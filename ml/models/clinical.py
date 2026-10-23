"""Clinical risk-factor model (multinomial logistic regression) + score fusion.

Stored as plain JSON numbers, not a pickle: loading it can never execute
code, and every parameter can be printed and explained.

    z      = (x - mean) / scale                (standardise each feature)
    logits = z @ coef.T + intercept            (3 numbers)
    scores = softmax(logits)

Fusion with the image model (a weighted geometric mean, renormalised):

    fused_c  proportional to  clinical_c^(1 - w) * image_c^w
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ml.data.clinical import CLINICAL_FEATURES

DEFAULT_C = 0.1  # chosen on validation folds (best ROC-AUC of 0.01 / 0.1 / 1.0)
DEFAULT_IMAGE_WEIGHT = 0.2  # chosen on validation folds (docs/model_training.md)
# Screening decision rule, chosen on validation folds: the label is Osteoporosis
# whenever 1.2 x its score beats every other score. Raised validation
# osteoporosis recall 0.72 -> 0.79 at macro F1 0.61 -> 0.59.
DEFAULT_OSTEOPOROSIS_BOOST = 1.2


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)  # avoids overflow in exp()
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class ClinicalModel:
    features: list[str]
    classes: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[list[float]]  # (n_classes, n_features)
    intercept: list[float]
    C: float
    fusion_image_weight: float = DEFAULT_IMAGE_WEIGHT
    osteoporosis_boost: float = 1.0  # 1.0 = plain "highest score wins"
    note: str = ""

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, classes, C: float = DEFAULT_C, seed: int = 42,
            fusion_image_weight: float = DEFAULT_IMAGE_WEIGHT, osteoporosis_boost: float = 1.0,
            note: str = "") -> "ClinicalModel":
        from sklearn.linear_model import LogisticRegression

        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0  # a constant feature must not cause division by zero
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit((X - mean) / scale, y)
        if list(clf.classes_) != list(range(len(classes))):
            raise ValueError("Every class must be present in the training labels")
        return cls(features=list(CLINICAL_FEATURES), classes=list(classes), mean=mean.tolist(),
                   scale=scale.tolist(), coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist(), C=C,
                   fusion_image_weight=fusion_image_weight, osteoporosis_boost=osteoporosis_boost, note=note)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != len(self.features) or not np.isfinite(X).all():
            raise ValueError(f"Expected {len(self.features)} finite clinical features, got {X.shape}")
        z = (X - np.array(self.mean)) / np.array(self.scale)
        return softmax(z @ np.array(self.coef).T + np.array(self.intercept))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ClinicalModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(**data)
        if model.features != list(CLINICAL_FEATURES):
            raise ValueError("clinical_model.json was built for a different feature list")
        return model


def fuse_scores(clinical: np.ndarray, image: np.ndarray, image_weight: float) -> np.ndarray:
    """Weighted geometric mean of two score vectors, renormalised to sum to 1.

    Worked example, w = 0.2: clinical (0.2, 0.3, 0.5), image (0.3, 0.6, 0.1)
      0.2^0.8 * 0.3^0.2 = 0.2169
      0.3^0.8 * 0.6^0.2 = 0.3446
      0.5^0.8 * 0.1^0.2 = 0.3624      sum = 0.9239
      fused = (0.235, 0.373, 0.392)   -> still Osteoporosis, but less sure than clinical alone
    (checked in tests/test_clinical.py)
    """
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError("image_weight must be between 0 and 1")
    log_fused = (1.0 - image_weight) * np.log(np.clip(clinical, 1e-9, 1.0)) \
        + image_weight * np.log(np.clip(image, 1e-9, 1.0))
    return softmax(log_fused)


def screening_decision(scores: np.ndarray, class_names, osteoporosis_boost: float) -> int:
    """Index of the output label under the screening rule.

    Multiplies the Osteoporosis score by `osteoporosis_boost` before taking the
    largest. The scores themselves are NOT changed; only the choice of label.

    Example, boost 1.2: scores (0.10, 0.48, 0.42) -> 0.42 x 1.2 = 0.504 > 0.48,
    so the label is Osteoporosis even though Osteopenia has the higher score.
    """
    if osteoporosis_boost < 1.0:
        raise ValueError("osteoporosis_boost must be >= 1")
    weights = np.ones(len(class_names))
    weights[list(class_names).index("Osteoporosis")] = osteoporosis_boost
    return int(np.argmax(np.asarray(scores) * weights))
