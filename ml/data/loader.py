"""Turn manifest records into arrays and tf.data pipelines.

The ~210 preprocessed images take 210 x 224 x 224 x 3 x 4 bytes ~= 126 MB,
so we preprocess everything once, keep it in memory and cache it on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from config import PreprocessingConfig
from ml.data.manifest import ImageRecord
from ml.preprocessing import preprocess_image, read_image_file


def _cache_key(records: list[ImageRecord], cfg: PreprocessingConfig) -> str:
    """Hash of (preprocessing settings + exact files) — changes if either changes."""
    payload = json.dumps({"cfg": cfg.to_dict(), "files": [(r.path, r.sha256) for r in records]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_arrays(
    records: list[ImageRecord],
    dataset_dir: Path,
    cfg: PreprocessingConfig,
    class_names: tuple[str, ...],
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, H, W, 3) float32 in [0, 255] and y (N,) int class indices."""
    index_of = {name: i for i, name in enumerate(class_names)}
    y = np.array([index_of[r.label] for r in records], dtype=np.int32)

    cache = cache_dir / f"images_{_cache_key(records, cfg)}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache), y

    X = np.stack([preprocess_image(read_image_file(dataset_dir / r.path), cfg) for r in records])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return X, y


def class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """'Balanced' weights: n_samples / (n_classes * count_c).

    Example with counts [18, 82, 26] (126 images): Normal gets 126/(3*18)=2.33,
    Osteopenia 0.51, Osteoporosis 1.62 — a missed Normal costs ~4.5x a missed
    Osteopenia in the loss, counteracting the imbalance.
    """
    counts = np.bincount(y, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"A class is missing from the training labels: counts={counts.tolist()}")
    return {c: float(len(y) / (n_classes * counts[c])) for c in range(n_classes)}


def make_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool, seed: int, augmenter=None):
    """tf.data pipeline. Shuffling and augmentation happen ONLY when training=True."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augmenter is not None:
        ds = ds.map(lambda images, labels: (augmenter(images, training=True), labels),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
