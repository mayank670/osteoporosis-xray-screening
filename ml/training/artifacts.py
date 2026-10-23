"""Files saved next to every trained model, so the app never has to guess.

model directory/
    model.keras         weights + architecture + in-model normalisation
    class_map.json      {"0": "Normal", "1": "Osteopenia", "2": "Osteoporosis"}
    preprocessing.json  the exact PreprocessingConfig used
    metadata.json       model version, backbone, Grad-CAM layer, training config, environment
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

ENV_PACKAGES = ("tensorflow", "keras", "numpy", "opencv-python-headless", "scikit-learn")


def environment_info() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ENV_PACKAGES}}


def write_model_metadata(
    model_dir: Path, class_names, prep_cfg: dict, train_cfg: dict, model_version: str, extra: dict | None = None
) -> None:
    from ml.models import BACKBONES

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "class_map.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(class_names)}, indent=1), encoding="utf-8")
    (model_dir / "preprocessing.json").write_text(json.dumps(prep_cfg, indent=1), encoding="utf-8")
    metadata = {
        "model_version": model_version,
        "backbone": train_cfg["backbone"],
        "gradcam_layer": BACKBONES[train_cfg["backbone"]].gradcam_layer,
        "training_config": train_cfg,
        "environment": environment_info(),
        **(extra or {}),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
