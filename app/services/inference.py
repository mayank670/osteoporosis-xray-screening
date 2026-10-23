"""Holds the single Predictor instance for the whole web application.

The model is loaded ONCE at startup (several seconds, ~100 MB of RAM), not per
request. If loading fails, the app still starts and every page explains how to
install a model, instead of crashing or silently using an untrained model.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """No usable model is loaded."""


class InferenceService:
    def __init__(self) -> None:
        self.predictor = None
        self.error: str | None = "Model not loaded yet."
        self.model_dir: Path | None = None
        self._lock = threading.Lock()  # one prediction at a time: bounded memory on small machines

    def load(self, model_dir: Path) -> None:
        from ml.inference import ModelArtifactError, Predictor

        self.model_dir = Path(model_dir)
        try:
            self.predictor = Predictor.from_directory(self.model_dir)
            self.error = None
            log.info("Loaded model %s from %s", self.predictor.model_version, self.model_dir)
        except ModelArtifactError as exc:
            self.predictor = None
            self.error = str(exc)
            log.error("Model unavailable: %s", exc)

    @property
    def ready(self) -> bool:
        return self.predictor is not None

    @property
    def model_version(self) -> str | None:
        return self.predictor.model_version if self.predictor else None

    @property
    def metadata(self) -> dict:
        return self.predictor.metadata if self.predictor else {}

    def predict(self, image, **options):
        if self.predictor is None:
            raise ModelUnavailableError(self.error or "Model not loaded.")
        with self._lock:
            return self.predictor.predict(image, **options)


inference_service = InferenceService()
