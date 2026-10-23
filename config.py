"""Central configuration shared by training, evaluation and the web app.

Machine-specific values (folders, secrets) come from environment variables so
nothing personal is hard-coded. Experiment-defining values (class names, image
size) live here so that training and inference always read the same numbers.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Read KEY=VALUE pairs from .env (if it exists) into os.environ.
# Variables already set in the real environment take priority.
load_dotenv(PROJECT_ROOT / ".env")


def path_from_env(name: str, default: str) -> Path:
    """Return the folder named by environment variable `name`.

    Relative paths are resolved against the project root, so the project
    works no matter which directory the command is launched from.
    """
    path = Path(os.environ.get(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


# Keep Keras' downloaded ImageNet weights on the project drive instead of
# C:\Users\<you>\.keras. Keras reads KERAS_HOME when it is first imported, so
# this must run before any `import keras` (every entry point imports config first).
os.environ.setdefault("KERAS_HOME", str(path_from_env("KERAS_HOME", "artifacts/keras_home")))

DATASET_DIR = path_from_env("DATASET_DIR", "data/raw")
ARTIFACTS_DIR = path_from_env("ARTIFACTS_DIR", "artifacts")
UPLOADS_DIR = path_from_env("UPLOADS_DIR", "uploads")
REPORTS_DIR = path_from_env("REPORTS_DIR", "reports")

# Output neuron i of the model means CLASS_NAMES[i]. This order is saved next
# to every trained model; the app reads it from there instead of guessing.
CLASS_NAMES: tuple[str, ...] = ("Normal", "Osteopenia", "Osteoporosis")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings that change the pixels the model sees.

    frozen=True makes instances read-only, so a running experiment cannot
    accidentally change its own preprocessing halfway through.
    """

    image_size: tuple[int, int] = (224, 224)  # (height, width) fed to the model
    keep_aspect_ratio: bool = True  # pad to a square before resizing, never stretch
    use_clahe: bool = False  # off until an experiment shows it helps
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    version: str = "prep-v1"  # bump whenever any setting above changes meaning

    def to_dict(self) -> dict:
        """Plain dictionary, ready to be saved as JSON beside a trained model."""
        return asdict(self)


PREPROCESSING = PreprocessingConfig()


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters. These are starting values to validate, not proven optima."""

    backbone: str = "efficientnet_b0"
    seed: int = 42
    n_folds: int = 5
    batch_size: int = 16
    dropout: float = 0.3
    use_class_weights: bool = True
    # Phase A: frozen backbone, new head only
    head_learning_rate: float = 1e-3
    head_epochs: int = 40
    # Phase B: unfreeze the top of the backbone, much smaller steps
    finetune_learning_rate: float = 1e-5
    finetune_epochs: int = 30
    unfreeze_top_layers: int = 30  # counted from the end of the backbone
    # Model selection uses the VALIDATION set only
    monitor: str = "val_loss"
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    # Bit-exact reproducibility. Some GPU kernels have no deterministic version and
    # raise an error in this mode, so GPU experiments (Colab) switch it off; the seed
    # still fixes initial weights, shuffling and augmentation.
    deterministic_ops: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


TRAINING = TrainingConfig()

DATA_ARTIFACTS_DIR = ARTIFACTS_DIR / "data"  # manifest, folds, split report
MODEL_DIR = path_from_env("MODEL_DIR", "artifacts/models/active")  # model used by the web app
