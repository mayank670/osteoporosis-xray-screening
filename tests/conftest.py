"""Shared pytest fixtures. All images here are synthetic software-test data."""

import config  # noqa: F401  (sets KERAS_HOME on D: before any test imports keras)

import cv2
import numpy as np
import pytest


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode an array (gray, or RGB which we convert to OpenCV's BGR) to file bytes."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def gray8() -> np.ndarray:
    """300x200 (H x W) 8-bit horizontal gradient 0..255."""
    row = np.linspace(0, 255, 200).astype(np.uint8)
    return np.tile(row, (300, 1))


@pytest.fixture
def gray16() -> np.ndarray:
    """12-bit data stored in 16-bit, the common case for digital X-rays."""
    row = np.linspace(100, 4000, 200).astype(np.uint16)
    return np.tile(row, (300, 1))


@pytest.fixture
def rgb8(gray8) -> np.ndarray:
    return np.stack([gray8, gray8, gray8], axis=2)


@pytest.fixture
def png_bytes(gray8) -> bytes:
    return encode(gray8)


def make_xray_like(seed: int, size=(256, 256)) -> np.ndarray:
    """Random smooth uint8 texture: distinct per seed, NOT a real X-ray."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (size[0] // 8, size[1] // 8)).astype(np.uint8)
    return cv2.resize(noise, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """normal/ osteopenia/ osteoporosis/ with 10 distinct images each, plus planted duplicates."""
    root = tmp_path / "dataset"
    prefixes = {"normal": "N", "osteopenia": "OP", "osteoporosis": "OS"}
    seed = 0
    for folder, prefix in prefixes.items():
        (root / folder).mkdir(parents=True)
        for i in range(1, 11):
            seed += 1
            (root / folder / f"{prefix}{i}.png").write_bytes(encode(make_xray_like(seed)))
    # same-class duplicate: OP11 is a byte copy of OP1
    (root / "osteopenia" / "OP11.png").write_bytes((root / "osteopenia" / "OP1.png").read_bytes())
    # cross-class conflict: OS11 is a byte copy of N1
    (root / "osteoporosis" / "OS11.png").write_bytes((root / "normal" / "N1.png").read_bytes())
    # near-duplicate: OP12 is OP2 resized and re-encoded as JPEG (different bytes)
    op2 = cv2.imread(str(root / "osteopenia" / "OP2.png"), cv2.IMREAD_GRAYSCALE)
    (root / "osteopenia" / "OP12.jpg").write_bytes(encode(cv2.resize(op2, (200, 200)), ".jpg"))
    # unreadable file with an image extension
    (root / "normal" / "N99.png").write_bytes(b"this is not a png")
    return root


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A small model with the SAME layer contract as the real classifier.

    Untrained (random weights) — used only to test software plumbing.
    """
    import json

    import keras

    keras.utils.set_random_seed(0)
    inputs = keras.Input((224, 224, 3), name="image")
    x = keras.layers.Rescaling(1 / 255.0)(inputs)
    x = keras.layers.Conv2D(8, 3, strides=4, activation="relu", name="top_conv")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    logits = keras.layers.Dense(3, name="logits")(x)
    model = keras.Model(inputs, keras.layers.Activation("softmax", name="scores")(logits))

    from config import PREPROCESSING

    directory = tmp_path_factory.mktemp("model")
    model.save(directory / "model.keras")
    (directory / "class_map.json").write_text(json.dumps({"0": "Normal", "1": "Osteopenia", "2": "Osteoporosis"}))
    (directory / "preprocessing.json").write_text(json.dumps(PREPROCESSING.to_dict()))
    (directory / "metadata.json").write_text(json.dumps(
        {"model_version": "tiny-test-model", "backbone": "tiny", "gradcam_layer": "top_conv",
         "heldout_performance": {"image": {"accuracy": 0.5, "macro_f1": 0.3, "roc_auc": 0.5,
                                           "osteoporosis_recall": 0.2, "n": 9},
                                 "hybrid": {"accuracy": 0.6, "macro_f1": 0.6, "roc_auc": 0.8,
                                            "osteoporosis_recall": 0.7, "n": 9}}}))

    # Clinical component fitted on RANDOM synthetic patients — plumbing only.
    from ml.models.clinical import ClinicalModel

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(30, 80, 60), rng.integers(0, 2, 60), rng.uniform(1.5, 1.8, 60),
                         rng.uniform(50, 90, 60), rng.uniform(18, 32, 60), rng.integers(0, 2, 60),
                         rng.uniform(0, 20, 60), *[rng.integers(0, 2, 60) for _ in range(6)], rng.uniform(0, 3, 60)])
    y = np.repeat([0, 1, 2], 20)
    ClinicalModel.fit(X, y, ("Normal", "Osteopenia", "Osteoporosis")).save(directory / "clinical_model.json")
    return directory


@pytest.fixture(scope="session")
def image_only_model_dir(tiny_model_dir, tmp_path_factory):
    """The tiny model WITHOUT clinical_model.json."""
    import shutil

    directory = tmp_path_factory.mktemp("model_image_only")
    for name in ("model.keras", "class_map.json", "preprocessing.json", "metadata.json"):
        shutil.copy(tiny_model_dir / name, directory / name)
    return directory
