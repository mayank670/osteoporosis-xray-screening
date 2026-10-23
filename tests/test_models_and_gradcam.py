import json
import shutil

import keras
import numpy as np
import pytest

from ml.explainability import GradCAM, GradCAMError, overlay_heatmap
from ml.inference import ModelArtifactError, Predictor
from ml.models import BACKBONES, build_model, set_trainable_top
from tests.conftest import make_xray_like

# ---------- real backbone contracts (random weights: no download, structure only) ----------

@pytest.fixture(scope="module")
def effnet():
    return build_model("efficientnet_b0", 3, weights=None)


def test_classifier_contract(effnet):
    model, _ = effnet
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 3)
    assert model.get_layer("top_conv").output.shape[1:3] == (7, 7)
    scores = model.predict(np.zeros((2, 224, 224, 3), np.float32), verbose=0)
    assert scores.shape == (2, 3) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)


def test_efficientnet_has_internal_normalisation(effnet):
    model, _ = effnet
    names = [type(layer).__name__ for layer in model.layers[:4]]
    assert "Rescaling" in names and "Normalization" in names


def test_freezing_keeps_batchnorm_frozen(effnet):
    _, base = effnet
    assert set_trainable_top(base, 0) == 0
    assert not any(layer.trainable for layer in base.layers)
    n = set_trainable_top(base, 30)
    trainable = [layer for layer in base.layers if layer.trainable]
    assert len(trainable) == n > 0
    assert not any(isinstance(layer, keras.layers.BatchNormalization) for layer in trainable)
    set_trainable_top(base, 0)


@pytest.mark.parametrize("name", ["resnet50", "densenet121", "vgg16"])
def test_benchmark_backbones_expose_gradcam_layer(name):
    model, _ = build_model(name, 3, weights=None)
    assert len(model.get_layer(BACKBONES[name].gradcam_layer).output.shape) == 4
    assert model.output_shape == (None, 3)


def test_caffe_preprocess_matches_keras_reference():
    from ml.models.backbones import CaffePreprocess

    x = np.random.default_rng(0).uniform(0, 255, (1, 8, 8, 3)).astype(np.float32)
    expected = keras.applications.resnet50.preprocess_input(x.copy())
    assert np.allclose(np.asarray(CaffePreprocess()(x)), expected, atol=1e-4)


def test_torch_preprocess_matches_keras_reference():
    from ml.models.backbones import TorchPreprocess

    x = np.random.default_rng(0).uniform(0, 255, (1, 8, 8, 3)).astype(np.float32)
    expected = keras.applications.densenet.preprocess_input(x.copy())
    assert np.allclose(np.asarray(TorchPreprocess()(x)), expected, atol=1e-4)

# ---------- Grad-CAM ----------

@pytest.fixture(scope="module")
def tiny_model(tiny_model_dir):
    return keras.models.load_model(tiny_model_dir / "model.keras")


@pytest.mark.parametrize("class_index", [0, 1, 2])
def test_heatmap_for_every_class(tiny_model, class_index):
    image = np.random.default_rng(class_index).uniform(0, 255, (224, 224, 3)).astype(np.float32)
    heatmap, informative = GradCAM(tiny_model, "top_conv").heatmap(image, class_index)
    assert heatmap.ndim == 2 and heatmap.dtype == np.float32
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert np.isfinite(heatmap).all()
    if informative:
        assert heatmap.max() == pytest.approx(1.0)


def test_gradcam_on_real_efficientnet_graph(effnet):
    model, _ = effnet
    heatmap, _ = GradCAM(model, "top_conv").heatmap(np.full((224, 224, 3), 128, np.float32), 2)
    assert heatmap.shape == (7, 7)


def test_gradcam_rejects_bad_inputs(tiny_model):
    cam = GradCAM(tiny_model, "top_conv")
    with pytest.raises(GradCAMError):
        cam.heatmap(np.zeros((224, 224, 3), np.float32), 5)
    with pytest.raises(GradCAMError):
        cam.heatmap(np.zeros((224, 224), np.float32), 0)
    with pytest.raises(GradCAMError, match="not found"):
        GradCAM(tiny_model, "no_such_layer")
    with pytest.raises(GradCAMError, match="must output"):
        GradCAM(tiny_model, "logits")


def test_overlay_shape_alpha_and_colormap():
    gray = np.full((100, 100), 50, np.uint8)
    heat = np.zeros((7, 7), np.float32)
    heat[3, 3] = 1.0
    assert overlay_heatmap(gray, heat, alpha=0.0).tolist() == np.stack([gray] * 3, axis=2).tolist()
    out = overlay_heatmap(gray, heat, alpha=0.5, colormap="inferno")
    assert out.shape == (100, 100, 3) and out.dtype == np.uint8
    with pytest.raises(ValueError):
        overlay_heatmap(gray, heat, alpha=1.5)
    with pytest.raises(ValueError):
        overlay_heatmap(gray, heat, colormap="rainbow-unicorn")

# ---------- Predictor (the production inference path) ----------

def test_predictor_end_to_end(tiny_model_dir):
    predictor = Predictor.from_directory(tiny_model_dir)
    out = predictor.predict(make_xray_like(1, (300, 260)), display_size=256)
    assert out.label in ("Normal", "Osteopenia", "Osteoporosis")
    assert set(out.scores) == {"Normal", "Osteopenia", "Osteoporosis"}
    assert sum(out.scores.values()) == pytest.approx(1.0, abs=1e-5)
    assert out.label == max(out.scores, key=out.scores.get)
    assert out.display_image.shape == (256, 256)
    assert out.overlay is None or out.overlay.shape == (256, 256, 3)
    assert predictor.model_version == "tiny-test-model"


def test_predictor_is_deterministic(tiny_model_dir):
    predictor = Predictor.from_directory(tiny_model_dir)
    image = make_xray_like(5)
    assert predictor.predict(image, explain=False).scores == predictor.predict(image, explain=False).scores


def test_missing_model_directory_gives_clear_error(tmp_path):
    with pytest.raises(ModelArtifactError, match="missing model.keras"):
        Predictor.from_directory(tmp_path / "nothing-here")


def test_class_map_mismatch_detected(tiny_model_dir, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(tiny_model_dir, broken)
    (broken / "class_map.json").write_text(json.dumps({"0": "A", "1": "B"}))
    with pytest.raises(ModelArtifactError, match="outputs"):
        Predictor.from_directory(broken)


def test_corrupt_model_file_detected(tiny_model_dir, tmp_path):
    broken = tmp_path / "corrupt"
    shutil.copytree(tiny_model_dir, broken)
    (broken / "model.keras").write_bytes(b"garbage")
    with pytest.raises(ModelArtifactError, match="Cannot load"):
        Predictor.from_directory(broken)
