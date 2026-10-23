"""Classification metrics with honest uncertainty estimates."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than p +- 1.96*SE for small n).

    Worked example: 6 of 8 osteoporosis cases found -> recall 0.75, but the
    interval is about (0.41, 0.93): with n=8 we cannot claim much.
    """
    if trials == 0:
        return (math.nan, math.nan)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, y_pred, metric, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval: resample cases with replacement, recompute."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        values.append(metric(y_true[idx], y_pred[idx]))
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, class_names, seed: int = 0,
                    y_pred: np.ndarray | None = None) -> dict:
    """y_true: (N,) int labels. y_score: (N, C) model scores. Returns a JSON-ready dict.

    y_pred: the chosen labels, if a decision rule other than "highest score"
    was used. Label metrics use y_pred; ROC-AUC always uses the raw scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim != 2 or y_score.shape != (len(y_true), len(class_names)):
        raise ValueError(f"y_score must be (N, {len(class_names)}), got {y_score.shape}")
    labels = list(range(len(class_names)))
    y_pred = y_score.argmax(axis=1) if y_pred is None else np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro_f1 = lambda t, p: f1_score(t, p, labels=labels, average="macro", zero_division=0)  # noqa: E731

    per_class = {}
    for i, name in enumerate(class_names):
        found = int(((y_pred == i) & (y_true == i)).sum())
        per_class[name] = {
            "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            "support": int(support[i]), "recall_95ci": wilson_interval(found, int(support[i])),
        }

    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_95ci": wilson_interval(int((y_pred == y_true).sum()), len(y_true)),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "macro_f1_95ci": bootstrap_ci(y_true, y_pred, macro_f1, seed=seed),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0, digits=3),
    }
    # ROC-AUC needs at least one positive and one negative case for every class.
    if all(0 < (y_true == i).sum() < len(y_true) for i in labels):
        if len(labels) == 2:  # binary: sklearn wants the positive-class score only
            result["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, y_score[:, 1]))
        else:
            result["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", labels=labels))
        result["roc_auc_per_class"] = {
            name: float(roc_auc_score((y_true == i).astype(int), y_score[:, i]))
            for i, name in enumerate(class_names)
        }
    else:
        result["roc_auc_macro_ovr"] = None
        result["roc_auc_note"] = "not computable: a class has no positive or no negative cases"
    return result
