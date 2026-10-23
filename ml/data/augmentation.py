"""Training-only random augmentation.

Kept OUT of the saved model on purpose: the inference model must be
deterministic, and the web app never needs these layers.

Choices are deliberately mild. Bone texture is the signal we hope the model
uses, so we avoid blur, noise or strong contrast changes that could erase it.
"""

from __future__ import annotations


def build_augmenter(seed: int):
    import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),  # left vs right knee
            keras.layers.RandomRotation(0.03, fill_mode="constant", seed=seed),  # +-~11 degrees
            keras.layers.RandomZoom(0.10, fill_mode="constant", seed=seed),
            keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed),
            keras.layers.RandomContrast(0.10, value_range=(0, 255), seed=seed),
        ],
        name="augmentation",
    )
