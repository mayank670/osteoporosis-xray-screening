"""Train one model on one fold: Phase A (head only) then Phase B (fine-tune).

The test subset is never passed in here. Model selection (early stopping,
checkpointing, learning-rate schedule) looks only at the validation subset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from config import PreprocessingConfig, TrainingConfig


def set_seeds(seed: int, deterministic_ops: bool = True) -> None:
    """Seed Python, NumPy and the Keras backend; optionally request deterministic TF ops.

    Works with any Keras backend (TensorFlow on the laptop, JAX on a Colab TPU);
    op determinism is a TensorFlow-only switch, so it is only touched when requested.
    """
    import keras

    keras.utils.set_random_seed(seed)  # seeds random, numpy and the active backend together
    if deterministic_ops:
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()


def _callbacks(cfg: TrainingConfig, checkpoint: Path, best_so_far: float | None, log_path: Path):
    import keras

    return [
        # Save only when val_loss beats everything seen so far — including Phase A.
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor=cfg.monitor, save_best_only=True,
            initial_value_threshold=best_so_far,
        ),
        keras.callbacks.EarlyStopping(
            monitor=cfg.monitor, patience=cfg.early_stopping_patience, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.monitor, factor=cfg.reduce_lr_factor, patience=cfg.reduce_lr_patience, min_lr=1e-7,
        ),
        keras.callbacks.CSVLogger(log_path, append=True),
    ]


def train_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: TrainingConfig, prep: PreprocessingConfig,
    class_names: tuple[str, ...], out_dir: Path,
    weights: str | None = "imagenet",
):
    """Train and save the best model to out_dir/model.keras. Returns (model, history dict)."""
    import keras

    from ml.data.augmentation import build_augmenter
    from ml.data.loader import class_weights, make_dataset
    from ml.models import build_model, set_trainable_top

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(cfg.seed, cfg.deterministic_ops)
    checkpoint = out_dir / "model.keras"
    log_path = out_dir / "training_log.csv"
    log_path.unlink(missing_ok=True)

    model, base = build_model(cfg.backbone, len(class_names), prep.image_size, cfg.dropout, weights)
    train_ds = make_dataset(X_train, y_train, cfg.batch_size, True, cfg.seed, build_augmenter(cfg.seed))
    val_ds = make_dataset(X_val, y_val, cfg.batch_size, False, cfg.seed)
    weights_by_class = class_weights(y_train, len(class_names)) if cfg.use_class_weights else None

    def compile_and_fit(learning_rate: float, epochs: int, initial_epoch: int, best: float | None):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return model.fit(
            train_ds, validation_data=val_ds, epochs=initial_epoch + epochs, initial_epoch=initial_epoch,
            class_weight=weights_by_class, callbacks=_callbacks(cfg, checkpoint, best, log_path), verbose=2,
        ).history

    # Phase A — backbone frozen; only the new Dense head learns.
    set_trainable_top(base, 0)
    history_a = compile_and_fit(cfg.head_learning_rate, cfg.head_epochs, 0, None)
    best_a = min(history_a[cfg.monitor])
    epochs_a = len(history_a["loss"])

    # Phase B — unfreeze the top of the backbone; tiny learning rate.
    history_b: dict = {}
    if cfg.finetune_epochs > 0 and cfg.unfreeze_top_layers > 0:
        set_trainable_top(base, cfg.unfreeze_top_layers)
        history_b = compile_and_fit(cfg.finetune_learning_rate, cfg.finetune_epochs, epochs_a, best_a)

    history = {
        key: [float(v) for v in history_a.get(key, [])] + [float(v) for v in history_b.get(key, [])]
        for key in ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate")
    }
    history["phase_a_epochs"] = epochs_a
    history["best_val_loss"] = float(min(v for v in history["val_loss"] if math.isfinite(v)))
    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    best_model = keras.models.load_model(checkpoint)  # the checkpointed best, not the last epoch
    return best_model, history
