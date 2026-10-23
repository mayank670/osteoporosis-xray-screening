"""Upload, prediction, results, history and private file downloads."""

from __future__ import annotations

import io

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required

from app.extensions import db
from app.forms import UploadForm
from app.models import Report, Scan
from app.services import storage
from app.services.inference import inference_service
from app.services.scans import ScanError, ensure_pdf, heatmap_array_path, process_upload
from ml.explainability import COLORMAPS, overlay_heatmap

bp = Blueprint("main", __name__)


def owned_scan_or_404(scan_id: int) -> Scan:
    """Fetch a scan that belongs to the logged-in user.

    Other users' scans give 404, not 403, so their existence is not revealed.
    """
    scan = db.session.get(Scan, scan_id)
    if scan is None or scan.user_id != current_user.id:
        abort(404)
    return scan


def _clinical_available() -> bool:
    return inference_service.ready and inference_service.predictor.supports_clinical


@bp.get("/health")
def health():
    return jsonify(status="ok", model_loaded=inference_service.ready,
                   model_version=inference_service.model_version, hybrid_available=_clinical_available())


@bp.get("/")
def index():
    return redirect(url_for("main.upload" if current_user.is_authenticated else "auth.login"))


@bp.get("/upload")
@login_required
def upload():
    return render_template("upload.html", form=UploadForm(), model_error=inference_service.error,
                           max_mb=request.max_content_length // (1024 * 1024),
                           clinical_available=_clinical_available())


@bp.post("/predict")
@login_required
def predict():
    form = UploadForm()
    wants_json = request.accept_mimetypes.best == "application/json"
    if not form.validate_on_submit():
        message = "; ".join(e for errors in form.errors.values() for e in errors) or "Invalid upload."
        if wants_json:
            return jsonify(error=message), 400
        return render_template("upload.html", form=form, model_error=inference_service.error,
                               max_mb=request.max_content_length // (1024 * 1024),
                               clinical_available=_clinical_available()), 400

    file = form.image.data
    try:
        scan = process_upload(current_user, file.read(), file.filename, form.clinical_inputs())
    except ScanError as exc:
        if wants_json:
            return jsonify(error=exc.message), exc.status
        form.image.errors.append(exc.message)
        return render_template("upload.html", form=form, model_error=inference_service.error,
                               max_mb=request.max_content_length // (1024 * 1024),
                               clinical_available=_clinical_available()), exc.status

    target = url_for("main.scan_detail", scan_id=scan.id)
    if wants_json:
        return jsonify(scan_id=scan.id, label=scan.prediction.label, scores=scan.prediction.scores,
                       mode=scan.prediction.mode, model_version=scan.prediction.model_version,
                       redirect=target), 201
    return redirect(target)


@bp.get("/history")
@login_required
def history():
    scans = (Scan.query.filter_by(user_id=current_user.id)
             .order_by(Scan.uploaded_at.desc()).limit(200).all())
    return render_template("history.html", scans=scans)


@bp.get("/scan/<int:scan_id>")
@login_required
def scan_detail(scan_id: int):
    scan = owned_scan_or_404(scan_id)
    return render_template("result.html", scan=scan, colormaps=sorted(COLORMAPS))


@bp.get("/scan/<int:scan_id>/image/<kind>")
@login_required
def scan_image(scan_id: int, kind: str):
    scan = owned_scan_or_404(scan_id)
    report = scan.prediction.report if scan.prediction else None
    relative = {"input": scan.display_path, "overlay": report.heatmap_path if report else None}.get(kind)
    if not relative:
        abort(404)
    path = storage.resolve_inside(current_app.config["UPLOADS_DIR"], relative)
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/png")


@bp.get("/scan/<int:scan_id>/overlay")
@login_required
def scan_overlay_custom(scan_id: int):
    """Re-colour the stored heatmap with user-chosen transparency and colour map."""
    scan = owned_scan_or_404(scan_id)
    report = scan.prediction.report if scan.prediction else None
    if not report or not report.heatmap_path or not scan.display_path:
        abort(404)
    try:
        alpha = float(request.args.get("alpha", "0.4"))
    except ValueError:
        abort(400)
    colormap = request.args.get("colormap", "jet")
    if not 0.0 <= alpha <= 1.0 or colormap not in COLORMAPS:
        abort(400)
    heatmap_file = heatmap_array_path(report.heatmap_path)
    display_file = storage.resolve_inside(current_app.config["UPLOADS_DIR"], scan.display_path)
    if not heatmap_file.is_file() or not display_file.is_file():
        abort(404)
    display = cv2.imread(str(display_file), cv2.IMREAD_GRAYSCALE)
    rgb = overlay_heatmap(display, np.load(heatmap_file), alpha=alpha, colormap=colormap)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        abort(500)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/png")


@bp.get("/report/<int:report_id>")
@login_required
def report_pdf(report_id: int):
    report = db.session.get(Report, report_id)
    if report is None or report.prediction.scan.user_id != current_user.id:
        abort(404)
    path = ensure_pdf(report)
    return send_file(path, mimetype="application/pdf", as_attachment=True,
                     download_name=f"osteo-report-R{report.id:06d}.pdf")

