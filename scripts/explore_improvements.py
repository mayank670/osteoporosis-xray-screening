"""Search for a more accurate model using VALIDATION folds only.

    python scripts/explore_improvements.py

Every candidate = features -> StandardScaler -> class-balanced logistic
regression, fitted on each fold's TRAIN part, scored on its VALIDATION part
(same 5 folds as everything else). Test parts are never touched.

Candidates:
  image:    EfficientNet-B0 frozen features at 224 / 448 / 640 px, and a
            3x3 grid of 224-px tiles from a 672-px image (averaged)
  texture:  Local Binary Pattern histograms at 3 scales (trabecular texture)
  clinical: age, sex, height, weight, BMI, menopause, risk factors
            (T-score, Z-score and Diagnosis are EXCLUDED: they are the label)
  combos:   image + clinical, texture + clinical

Output: artifacts/experiments/improvements_val.{json,md}
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.clinical import CLINICAL_FEATURES, load_clinical_table
from ml.data.loader import load_arrays
from ml.data.manifest import load_manifest
from ml.data.splits import load_parts, make_fold
from ml.preprocessing import read_image_file
from ml.preprocessing.transforms import letterbox_resize, to_grayscale, to_uint8

C_VALUES = (0.01, 0.1, 1.0)


def effnet_features(X: np.ndarray) -> np.ndarray:
    import keras

    from ml.models import build_model

    model, _ = build_model("efficientnet_b0", 3, X.shape[1:3], weights="imagenet")
    extractor = keras.Model(model.input, model.get_layer("head_pool").output)
    return extractor.predict(X, batch_size=8, verbose=0)


def tile_features(records, size=672, tile=224) -> np.ndarray:
    prep = dataclasses.replace(config.PREPROCESSING, image_size=(size, size))
    X, _ = load_arrays(records, config.DATASET_DIR, prep, config.CLASS_NAMES, config.ARTIFACTS_DIR / "cache")
    n = size // tile
    tiles = X.reshape(len(X), n, tile, n, tile, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, tile, tile, 3)
    feats = effnet_features(tiles).reshape(len(X), n * n, -1)
    return np.concatenate([feats.mean(axis=1), feats.max(axis=1)], axis=1)


def lbp_histogram(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """8-neighbour, radius-1 LBP codes (0..255) -> normalised histogram inside mask."""
    c = gray[1:-1, 1:-1].astype(np.int16)
    code = np.zeros_like(c, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx].astype(np.int16)
        code |= ((neighbour >= c).astype(np.uint8) << bit)
    hist = np.bincount(code[mask[1:-1, 1:-1]], minlength=256).astype(np.float64)
    return hist / max(1.0, hist.sum())


def texture_features(records) -> np.ndarray:
    rows = []
    for r in records:
        gray = to_uint8(to_grayscale(read_image_file(config.DATASET_DIR / r.path)))
        parts = []
        for side in (1024, 512, 256):
            g = letterbox_resize(gray, (side, side))
            parts.append(lbp_histogram(g, g > 40))  # ignore black background
        rows.append(np.concatenate(parts))
    return np.array(rows)


def evaluate(features: np.ndarray, y, index_of, records, parts, n_folds, seed, C: float) -> dict:
    true, score = [], []
    for fold in range(n_folds):
        split = make_fold(records, parts, fold, n_folds)
        tr = [index_of[r.patient_id] for r in split.train]
        va = [index_of[r.patient_id] for r in split.val]
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000, random_state=seed))
        clf.fit(features[tr], y[tr])
        true.append(y[va]); score.append(clf.predict_proba(features[va]))
    t, s = np.concatenate(true), np.concatenate(score)
    p = s.argmax(axis=1)
    osteo = config.CLASS_NAMES.index("Osteoporosis")
    return {"macro_f1": f1_score(t, p, average="macro"), "balanced_acc": balanced_accuracy_score(t, p),
            "accuracy": float((t == p).mean()), "auc": roc_auc_score(t, s, multi_class="ovr"),
            "osteo_recall": float(((p == osteo) & (t == osteo)).sum() / (t == osteo).sum())}


def main() -> None:
    data_dir = config.DATA_ARTIFACTS_DIR
    records = [r for r in load_manifest(data_dir / "manifest.csv") if not r.excluded]
    parts, n_folds, seed = load_parts(data_dir / "folds.json")
    index_of = {r.patient_id: i for i, r in enumerate(records)}
    y = np.array([config.CLASS_NAMES.index(r.label) for r in records])
    cache = config.ARTIFACTS_DIR / "cache"

    feature_sets: dict[str, np.ndarray] = {}
    clinical = load_clinical_table(config.DATASET_DIR)
    feature_sets["clinical only"] = np.array([clinical[r.patient_id] for r in records])
    for size in (224, 448, 640):
        t0 = time.time()
        prep = dataclasses.replace(config.PREPROCESSING, image_size=(size, size))
        X, _ = load_arrays(records, config.DATASET_DIR, prep, config.CLASS_NAMES, cache)
        feature_sets[f"image effnet {size}px"] = effnet_features(X)
        del X
        print(f"effnet {size}px features in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    feature_sets["image effnet 3x3 tiles @672"] = tile_features(records)
    print(f"tiles in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    feature_sets["texture LBP 3 scales"] = texture_features(records)
    print(f"LBP in {time.time() - t0:.0f}s", flush=True)

    image_keys = [k for k in feature_sets if k.startswith(("image", "texture"))]
    for key in image_keys:
        feature_sets[f"{key} + clinical"] = np.hstack([feature_sets[key], feature_sets["clinical only"]])

    results = {}
    for name, feats in feature_sets.items():
        results[name] = {f"C={C}": evaluate(feats, y, index_of, records, parts, n_folds, seed, C) for C in C_VALUES}
        best = max(results[name].values(), key=lambda m: m["macro_f1"])
        print(f"{name:<44} best macroF1 {best['macro_f1']:.3f}  AUC {best['auc']:.3f}  "
              f"osteo recall {best['osteo_recall']:.3f}", flush=True)

    out = config.ARTIFACTS_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "improvements_val.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["# Improvement search — validation folds only (n = 210 pooled)", "",
             "| Features | C | Macro F1 | Balanced acc. | Accuracy | ROC-AUC | Osteoporosis recall |",
             "|---|---|---|---|---|---|---|"]
    for name, by_c in results.items():
        for c, m in by_c.items():
            lines.append(f"| {name} | {c[2:]} | {m['macro_f1']:.3f} | {m['balanced_acc']:.3f} | "
                         f"{m['accuracy']:.3f} | {m['auc']:.3f} | {m['osteo_recall']:.3f} |")
    (out / "improvements_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {out / 'improvements_val.md'}")


if __name__ == "__main__":
    main()
