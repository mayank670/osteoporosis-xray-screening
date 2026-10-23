"""Load a trained model directory and run prediction + Grad-CAM.

Two modes:
  image_only  X-ray CNN alone (near chance on held-out data — see docs)
  hybrid      X-ray CNN fused with the clinical risk-factor model
              (used only when patient details are supplied AND the model
              directory contains clinical_model.json)

Independent of Flask, so it can be tested and benchmarked on its own. The web
app's service layer wraps this class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import PreprocessingConfig
from ml.explainability import GradCAM, GradCAMError, overlay_heatmap
from ml.models.clinical import ClinicalModel, fuse_scores, screening_decision
from ml.preprocessing import prepare_display_image, preprocess_image

REQUIRED_FILES = ("model.keras", "class_map.json", "preprocessing.json", "metadata.json")
OPTIONAL_FILES = ("clinical_model.json",)


class ModelArtifactError(RuntimeError):
    """The model directory is missing, incomplete or inconsistent."""


@dataclass
class PredictionOutput:
    label: str
    class_index: int
    scores: dict[str, float]  # final model scores — NOT calibrated probabilities
    mode: str  # "image_only" or "hybrid"
    image_scores: dict[str, float]
    clinical_scores: dict[str, float] | None
    osteoporosis_boost: float  # 1.0 = the label is simply the highest score
    display_image: np.ndarray  # (S, S) uint8, same geometry as the model input
    overlay: np.ndarray | None  # (S, S, 3) uint8 RGB Grad-CAM overlay
    heatmap: np.ndarray | None  # (h, w) float32 in [0, 1], model-input coordinates
    heatmap_informative: bool
    heatmap_error: str | None = None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"Cannot read {path.name}: {exc}") from exc


class Predictor:
    def __init__(self, model, class_names: tuple[str, ...], preprocessing: PreprocessingConfig, metadata: dict,
                 clinical_model: ClinicalModel | None = None):
        self.model = model
        self.clinical_model = clinical_model
        self.class_names = class_names
        self.preprocessing = preprocessing
        self.metadata = metadata
        self.model_version: str = metadata.get("model_version", "unknown")
        expected = (None, *preprocessing.image_size, 3)
        if tuple(model.input_shape) != expected:
            raise ModelArtifactError(f"Model input {model.input_shape} does not match preprocessing {expected}")
        if model.output_shape[-1] != len(class_names):
            raise ModelArtifactError(
                f"Model has {model.output_shape[-1]} outputs but class_map.json lists {len(class_names)}")
        # A model reloaded from .keras records its input as a one-element list,
        # and Keras warns unless the call passes inputs in that same structure.
        self._wrap_input = isinstance(model.input, list)
        self._gradcam = GradCAM(model, metadata["gradcam_layer"]) if metadata.get("gradcam_layer") else None
        if clinical_model is not None and tuple(clinical_model.classes) != tuple(class_names):
            raise ModelArtifactError("clinical_model.json classes do not match class_map.json")

    @property
    def supports_clinical(self) -> bool:
        return self.clinical_model is not None

    @classmethod
    def from_directory(cls, model_dir: Path) -> "Predictor":
        model_dir = Path(model_dir)
        missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
        if missing:
            raise ModelArtifactError(
                f"Model directory {model_dir} is missing {', '.join(missing)}. "
                "Train a model and promote it (see README: 'Training').")

        class_map = _load_json(model_dir / "class_map.json")
        try:
            class_names = tuple(class_map[str(i)] for i in range(len(class_map)))
        except KeyError as exc:
            raise ModelArtifactError("class_map.json must have keys '0'..'N-1'") from exc
        prep_dict = _load_json(model_dir / "preprocessing.json")
        preprocessing = PreprocessingConfig(**{k: tuple(v) if isinstance(v, list) else v
                                               for k, v in prep_dict.items()})
        metadata = _load_json(model_dir / "metadata.json")

        import keras

        import ml.models  # noqa: F401  registers custom layers

        try:
            model = keras.models.load_model(model_dir / "model.keras")
        except Exception as exc:  # Keras raises many exception types for bad files
            raise ModelArtifactError(f"Cannot load model.keras: {exc}") from exc

        clinical_model = None
        if (model_dir / "clinical_model.json").is_file():
            try:
                clinical_model = ClinicalModel.load(model_dir / "clinical_model.json")
            except (OSError, ValueError, TypeError) as exc:
                raise ModelArtifactError(f"Cannot load clinical_model.json: {exc}") from exc
        return cls(model, class_names, preprocessing, metadata, clinical_model)

    def predict(self, image: np.ndarray, clinical_features: np.ndarray | None = None, display_size: int = 512,
                alpha: float = 0.4, colormap: str = "jet", explain: bool = True) -> PredictionOutput:
        """image: decoded array from ml.preprocessing.decode_image.

        clinical_features: vector from ml.data.clinical.encode_clinical, or None
        for an X-ray-only prediction.
        """
        model_input = preprocess_image(image, self.preprocessing)
        batch = model_input[np.newaxis]
        image_scores = np.asarray(self.model([batch] if self._wrap_input else batch, training=False))[0]
        if image_scores.shape != (len(self.class_names),) or not np.isfinite(image_scores).all():
            raise RuntimeError(f"Model returned invalid scores: {image_scores!r}")

        clinical_scores = None
        if clinical_features is not None:
            if self.clinical_model is None:
                raise ModelArtifactError("This model has no clinical component; omit patient details.")
            clinical_scores = self.clinical_model.predict_proba(clinical_features)[0]
            scores = fuse_scores(clinical_scores, image_scores, self.clinical_model.fusion_image_weight)
            boost = self.clinical_model.osteoporosis_boost
        else:
            scores = image_scores
            boost = 1.0  # the rule was tuned for the hybrid mode only
        class_index = screening_decision(scores, self.class_names, boost)
        display = prepare_display_image(image, display_size, self.preprocessing)

        overlay, heatmap, informative, error = None, None, False, None
        if explain and self._gradcam is not None:
            # The heatmap explains the IMAGE model's score for the final class.
            try:
                heatmap, informative = self._gradcam.heatmap(model_input, class_index)
                overlay = overlay_heatmap(display, heatmap, alpha, colormap)
            except GradCAMError as exc:
                error = str(exc)
        return PredictionOutput(
            label=self.class_names[class_index],
            class_index=class_index,
            scores={name: float(s) for name, s in zip(self.class_names, scores)},
            mode="hybrid" if clinical_scores is not None else "image_only",
            image_scores={name: float(s) for name, s in zip(self.class_names, image_scores)},
            clinical_scores=None if clinical_scores is None
            else {name: float(s) for name, s in zip(self.class_names, clinical_scores)},
            osteoporosis_boost=boost,
            display_image=display,
            overlay=overlay,
            heatmap=heatmap,
            heatmap_informative=informative,
            heatmap_error=error,
        )
