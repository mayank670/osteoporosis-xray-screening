import numpy as np
import pytest

from ml.evaluation.metrics import compute_metrics, wilson_interval

CLASSES = ("Normal", "Osteopenia", "Osteoporosis")


def one_hot_scores(pred, n_classes=3):
    scores = np.full((len(pred), n_classes), 0.1)
    scores[np.arange(len(pred)), pred] = 0.8
    return scores


def test_wilson_worked_example():
    low, high = wilson_interval(6, 8)
    assert low == pytest.approx(0.409, abs=0.01) and high == pytest.approx(0.929, abs=0.01)


def test_wilson_edges():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert all(np.isnan(wilson_interval(0, 0)))


def test_perfect_predictions():
    y = np.array([0, 1, 2, 1, 2, 0])
    m = compute_metrics(y, one_hot_scores(y), CLASSES)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert m["roc_auc_macro_ovr"] == 1.0


def test_known_confusion_and_recall():
    y_true = np.array([2, 2, 2, 2, 1, 1, 0, 0])
    y_pred = np.array([2, 2, 1, 1, 1, 1, 0, 1])
    m = compute_metrics(y_true, one_hot_scores(y_pred), CLASSES)
    assert m["per_class"]["Osteoporosis"]["recall"] == 0.5
    assert m["per_class"]["Osteoporosis"]["support"] == 4
    assert m["per_class"]["Osteopenia"]["precision"] == pytest.approx(2 / 5)
    assert m["accuracy"] == pytest.approx(5 / 8)


def test_auc_skipped_when_class_absent():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, one_hot_scores(y), CLASSES)
    assert m["roc_auc_macro_ovr"] is None and "not computable" in m["roc_auc_note"]


def test_bad_score_shape_rejected():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.zeros((2, 2)), CLASSES)


def test_binary_metrics_and_auc():
    y = np.array([0, 0, 1, 1])
    scores = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]])
    m = compute_metrics(y, scores, ("Normal", "Osteoporosis"))
    assert m["accuracy"] == 1.0 and m["roc_auc_macro_ovr"] == 1.0
