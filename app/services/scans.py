"""The prediction workflow: validate -> store -> predict -> explain -> record -> report."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
from flask import current_app

from app.extensions import db
from app.models import Prediction, Report, Scan, User, utcnow
from app.services import storage
from app.services.inference import ModelUnavailableError, inference_service
from app.services.reports import build_pdf
from ml.data.clinical import encode_clinical
from ml.preprocessing import InvalidImageError, decode_image
from ml.preprocessing.image_io import detect_format

log = logging.getLogger(__name__)


class ScanError(Exception):
    """A user-facing failure. `message` is safe to show in the browser."""

    def __init__(self, message: str, status: int = 400, scan: Scan | None = None):
        super().__init__(message)
        self.message, self.status, self.scan = message, status, scan


def _uploads() -> Path:
    return Path(current_app.config["UPLOADS_DIR"])


def _reports() -> Path:
    return Path(current_app.config["REPORTS_DIR"])


def clinical_vector(inputs: dict):
    """Form values -> model feature vector (height converted from cm to m)."""
    return encode_clinical(
        age=inputs["age"], sex=inputs["sex"], height_m=inputs["height_cm"] / 100.0,
        weight_kg=inputs["weight_kg"], menopause_age=inputs.get("menopause_age"),
        smoker=inputs.get("smoker", False), diabetic=inputs.get("diabetic", False),
        hypothyroidism=inputs.get("hypothyroidism", False), estrogen_use=inputs.get("estrogen_use", False),
        fracture_history=inputs.get("fracture_history", False), family_history=inputs.get("family_history", False),
        walking_km=inputs.get("walking_km"),
    )


def process_upload(user: User, data: bytes, original_filename: str | None,
                   clinical_inputs: dict | None = None) -> Scan:
    """Run the full pipeline for one uploaded file and return the saved Scan.

    clinical_inputs (optional) switches from the X-ray-only model to the
    hybrid X-ray + patient-details model.
    """
    if not inference_service.ready:
        raise ScanError("The analysis model is not installed. Please contact the administrator.", 503)
    if clinical_inputs is not None and not inference_service.predictor.supports_clinical:
        raise ScanError("The installed model has no clinical component; untick 'Add patient details'.", 400)

    # 1. Validate the CONTENT (not the filename) before anything is stored.
    try:
        image = decode_image(data)  # checks emptiness, signature, decodability and size
        fmt = detect_format(data)
    except InvalidImageError as exc:
        raise ScanError(str(exc), 400) from exc

    # 2. Store the original under a random name and create the Scan row.
    extension = {"jpeg": "jpg", "tiff": "tif"}.get(fmt, fmt)
    scan = Scan(user_id=user.id, status="processing",
                image_path=storage.save_bytes(_uploads(), "originals", data, extension),
                original_filename=(original_filename or "upload")[:255])
    db.session.add(scan)
    db.session.commit()

    try:
        # 3-5. Preprocess, predict and explain (inside Predictor).
        features = clinical_vector(clinical_inputs) if clinical_inputs is not None else None
        result = inference_service.predict(image, clinical_features=features)
        scan.display_path = storage.save_png(_uploads(), "display", result.display_image)
        heatmap_path = None
        if result.overlay is not None:
            heatmap_path = storage.save_png(_uploads(), "overlays", result.overlay)
            # Raw heatmap beside the overlay, so the UI can re-colour it without re-running the model.
            np.save(heatmap_array_path(heatmap_path), result.heatmap)

        # 6. Record the prediction.
        prediction = Prediction(
            scan=scan, label=result.label, confidence=result.scores[result.label],
            score_normal=result.scores["Normal"], score_osteopenia=result.scores["Osteopenia"],
            score_osteoporosis=result.scores["Osteoporosis"],
            model_version=inference_service.model_version,
            preprocessing_version=inference_service.predictor.preprocessing.version,
            heatmap_informative=result.heatmap_informative,
            mode=result.mode,
            clinical_inputs=json.dumps(clinical_inputs) if clinical_inputs is not None else None,
            component_scores=json.dumps({"image": result.image_scores, "clinical": result.clinical_scores,
                                         "osteoporosis_boost": result.osteoporosis_boost}),
        )
        report = Report(prediction=prediction, heatmap_path=heatmap_path)
        db.session.add_all([prediction, report])
        db.session.flush()  # assigns report.id without committing yet

        # 7. PDF (failure here does not lose the prediction; it is regenerated on download).
        try:
            report.pdf_path = generate_pdf(report)
        except Exception:  # ReportLab can raise many types; log and continue
            log.exception("PDF generation failed for scan %s", scan.id)

        scan.status = "completed"
        db.session.commit()
        return scan
    except ModelUnavailableError as exc:
        _fail(scan, "The analysis model is not available.")
        raise ScanError("The analysis model is not available.", 503, scan) from exc
    except Exception as exc:
        log.exception("Analysis failed for scan %s", scan.id)  # full details in the server log only
        _fail(scan, "The image could not be analysed.")
        raise ScanError("The image could not be analysed. Please try another image.", 500, scan) from exc


def _fail(scan: Scan, message: str) -> None:
    db.session.rollback()
    scan = db.session.get(Scan, scan.id)
    scan.status, scan.error_message = "failed", message
    db.session.commit()


def heatmap_array_path(heatmap_path: str) -> Path:
    return storage.resolve_inside(_uploads(), heatmap_path).with_suffix(".npy")


def generate_pdf(report: Report) -> str:
    prediction = report.prediction
    scan = prediction.scan
    relative = f"pdf/{storage.new_name('pdf')}"
    target = storage.resolve_inside(_reports(), relative)

    def optional(relative_path):
        return storage.resolve_inside(_uploads(), relative_path) if relative_path else None

    build_pdf(target, report_id=report.id, scan_id=scan.id, predicted_at=prediction.predicted_at or utcnow(),
              label=prediction.label, scores=prediction.scores, model_version=prediction.model_version,
              preprocessing_version=prediction.preprocessing_version,
              original_image=optional(scan.display_path), overlay_image=optional(report.heatmap_path),
              heatmap_informative=prediction.heatmap_informative, mode=prediction.mode,
              component_scores=prediction.component_scores_dict, clinical_inputs=prediction.clinical_inputs_dict)
    return relative


def ensure_pdf(report: Report) -> Path:
    """Return the PDF path, regenerating it if the file is missing."""
    if report.pdf_path:
        path = storage.resolve_inside(_reports(), report.pdf_path)
        if path.is_file():
            return path
    report.pdf_path = generate_pdf(report)
    db.session.commit()
    return storage.resolve_inside(_reports(), report.pdf_path)


def delete_scan(scan: Scan) -> None:
    """Delete a scan's files and rows (used by the retention clean-up)."""
    storage.delete_quietly(_uploads(), scan.image_path)
    storage.delete_quietly(_uploads(), scan.display_path)
    if scan.prediction and scan.prediction.report:
        heatmap_path = scan.prediction.report.heatmap_path
        storage.delete_quietly(_uploads(), heatmap_path)
        if heatmap_path:
            storage.delete_quietly(_uploads(), heatmap_path.removesuffix(".png") + ".npy")
        storage.delete_quietly(_reports(), scan.prediction.report.pdf_path)
    db.session.delete(scan)


def purge_older_than(days: int) -> int:
    cutoff = utcnow() - timedelta(days=days)
    old = Scan.query.filter(Scan.uploaded_at < cutoff).all()
    for scan in old:
        delete_scan(scan)
    db.session.commit()
    return len(old)
