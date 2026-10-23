import numpy as np
import pytest

from ml.data.clinical import CLINICAL_FEATURES, encode_clinical, load_clinical_table
from ml.models.clinical import ClinicalModel, fuse_scores, softmax

CLASSES = ("Normal", "Osteopenia", "Osteoporosis")


def test_encode_clinical_order_and_derived_values():
    v = encode_clinical(age=60, sex="female", height_m=1.6, weight_kg=64, menopause_age=48,
                        smoker=True, family_history=True, walking_km=2)
    f = dict(zip(CLINICAL_FEATURES, v))
    assert len(v) == len(CLINICAL_FEATURES)
    assert f["female"] == 1 and f["bmi"] == pytest.approx(64 / 1.6 ** 2)
    assert f["postmenopausal"] == 1 and f["years_since_menopause"] == 12
    assert f["smoker"] == 1 and f["family_history"] == 1 and f["diabetic"] == 0
    assert f["walking_km"] == 2


def test_menopause_ignored_for_male_and_missing():
    male = dict(zip(CLINICAL_FEATURES, encode_clinical(60, "male", 1.7, 70, menopause_age=50)))
    assert male["postmenopausal"] == 0 and male["years_since_menopause"] == 0
    pre = dict(zip(CLINICAL_FEATURES, encode_clinical(40, "female", 1.6, 60, menopause_age=None)))
    assert pre["postmenopausal"] == 0


def test_label_columns_are_never_features():
    for forbidden in ("t_score", "z_score", "diagnosis"):
        assert not any(forbidden in name for name in CLINICAL_FEATURES)


def test_softmax_is_stable_for_large_logits():
    out = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isfinite(out).all() and out.sum() == pytest.approx(1.0)


def test_fusion_worked_example():
    fused = fuse_scores(np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.6, 0.1]), 0.2)
    assert fused == pytest.approx([0.235, 0.373, 0.392], abs=1e-3)


def test_fusion_extremes():
    c, i = np.array([0.1, 0.2, 0.7]), np.array([0.6, 0.3, 0.1])
    assert fuse_scores(c, i, 0.0) == pytest.approx(c)
    assert fuse_scores(c, i, 1.0) == pytest.approx(i)
    with pytest.raises(ValueError):
        fuse_scores(c, i, 1.5)


@pytest.fixture
def fitted():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, len(CLINICAL_FEATURES)))
    y = np.repeat([0, 1, 2], 30)
    X[y == 2, 0] += 2.0  # make "age" informative for class 2
    return ClinicalModel.fit(X, y, CLASSES, C=0.1), X, y


def test_json_model_matches_sklearn(fitted):
    from sklearn.linear_model import LogisticRegression

    model, X, y = fitted
    z = (X - X.mean(0)) / X.std(0)
    reference = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=42).fit(z, y)
    assert model.predict_proba(X) == pytest.approx(reference.predict_proba(z), abs=1e-6)


def test_save_load_round_trip(fitted, tmp_path):
    model, X, _ = fitted
    model.save(tmp_path / "c.json")
    assert ClinicalModel.load(tmp_path / "c.json").predict_proba(X) == pytest.approx(model.predict_proba(X))


def test_rejects_wrong_feature_count_and_nan(fitted):
    model, _, _ = fitted
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((1, 3)))
    bad = np.zeros((1, len(CLINICAL_FEATURES)))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.predict_proba(bad)


def test_fit_requires_every_class():
    X = np.random.default_rng(0).normal(size=(10, len(CLINICAL_FEATURES)))
    with pytest.raises(ValueError):
        ClinicalModel.fit(X, np.array([0, 1] * 5), CLASSES)


def test_predictor_hybrid_and_image_only(tiny_model_dir):
    from ml.inference import Predictor
    from tests.conftest import make_xray_like

    predictor = Predictor.from_directory(tiny_model_dir)
    image = make_xray_like(4)
    image_only = predictor.predict(image, explain=False)
    hybrid = predictor.predict(image, encode_clinical(70, "female", 1.5, 50, 45), explain=False)
    assert image_only.mode == "image_only" and image_only.clinical_scores is None
    assert hybrid.mode == "hybrid" and hybrid.image_scores == image_only.image_scores
    w = predictor.clinical_model.fusion_image_weight
    expected = fuse_scores(np.array(list(hybrid.clinical_scores.values())),
                           np.array(list(hybrid.image_scores.values())), w)
    assert list(hybrid.scores.values()) == pytest.approx(expected)


def test_real_spreadsheet_if_available():
    import config

    if not config.DATASET_DIR.is_dir():
        pytest.skip("dataset not downloaded")
    table = load_clinical_table(config.DATASET_DIR)
    assert len(table) == 240 and all(np.isfinite(v).all() for v in table.values())


def test_screening_decision_worked_example():
    from ml.models.clinical import screening_decision

    scores = np.array([0.10, 0.48, 0.42])
    assert screening_decision(scores, CLASSES, 1.0) == 1  # plain argmax: Osteopenia
    assert screening_decision(scores, CLASSES, 1.2) == 2  # 0.42 x 1.2 = 0.504 > 0.48
    assert screening_decision(np.array([0.10, 0.60, 0.30]), CLASSES, 1.2) == 1  # 0.36 < 0.60
    with pytest.raises(ValueError):
        screening_decision(scores, CLASSES, 0.5)


def test_predictor_applies_boost_only_in_hybrid_mode(tiny_model_dir, tmp_path):
    import json
    import shutil

    from ml.inference import Predictor
    from tests.conftest import make_xray_like

    boosted = tmp_path / "boosted"
    shutil.copytree(tiny_model_dir, boosted)
    data = json.loads((boosted / "clinical_model.json").read_text())
    data["osteoporosis_boost"] = 50.0  # extreme, so the effect is certain
    (boosted / "clinical_model.json").write_text(json.dumps(data))
    predictor = Predictor.from_directory(boosted)
    image = make_xray_like(2)
    hybrid = predictor.predict(image, encode_clinical(70, "female", 1.5, 50, 45), explain=False)
    assert hybrid.label == "Osteoporosis" and hybrid.osteoporosis_boost == 50.0
    assert sum(hybrid.scores.values()) == pytest.approx(1.0)  # scores untouched by the rule
    image_only = predictor.predict(image, explain=False)
    assert image_only.osteoporosis_boost == 1.0
    assert image_only.label == max(image_only.scores, key=image_only.scores.get)


def test_metrics_accept_explicit_predictions():
    from ml.evaluation.metrics import compute_metrics

    y = np.array([0, 1, 2, 2])
    scores = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.5, 0.4], [0.1, 0.2, 0.7]])
    plain = compute_metrics(y, scores, CLASSES)
    ruled = compute_metrics(y, scores, CLASSES, y_pred=np.array([0, 1, 2, 2]))
    assert plain["per_class"]["Osteoporosis"]["recall"] == 0.5
    assert ruled["per_class"]["Osteoporosis"]["recall"] == 1.0
    assert ruled["roc_auc_macro_ovr"] == plain["roc_auc_macro_ovr"]  # ranking unchanged
