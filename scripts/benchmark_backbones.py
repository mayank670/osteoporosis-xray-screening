"""Benchmark backbones (and CLAHE on/off) with frozen-feature linear probes.

    python scripts/benchmark_backbones.py

For each backbone we extract ImageNet features ONCE (frozen network, global
average pooling), then fit a class-balanced logistic regression on each fold's
TRAIN part and score its VALIDATION part. Same folds for every backbone.
The test parts are not used, so this comparison can guide decisions.

Also runs a "shortcut" baseline: logistic regression on trivial image
properties (original size, aspect ratio, file type, colour mode, brightness).
If that baseline scores as well as a CNN, the CNN may be using acquisition
artefacts instead of bone appearance.

Output: artifacts/benchmark/benchmark_val.json and .md
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.loader import load_arrays
from ml.data.manifest import load_manifest
from ml.data.splits import load_parts, make_fold
from ml.preprocessing import read_image_file


def extract_features(backbone: str, X: np.ndarray) -> np.ndarray:
    import keras

    from ml.models import build_model

    model, _ = build_model(backbone, 3, config.PREPROCESSING.image_size, weights="imagenet")
    extractor = keras.Model(model.input, model.get_layer("head_pool").output)
    return extractor.predict(X, batch_size=16, verbose=0)


def shortcut_features(records) -> np.ndarray:
    rows = []
    for r in records:
        path = config.DATASET_DIR / r.path
        image = read_image_file(path)
        h, w = image.shape[:2]
        rows.append([h, w, w / h, float(image.ndim == 3), float(path.suffix == ".JPEG"),
                     float(image.mean()), float(image.std()), path.stat().st_size / 1e6])
    return np.array(rows, dtype=np.float64)


def probe(features: np.ndarray, y: np.ndarray, index_of: dict, records, parts, n_folds, seed) -> dict:
    """Pooled validation predictions over all folds -> metrics."""
    true, pred, fold_f1 = [], [], []
    for fold in range(n_folds):
        split = make_fold(records, parts, fold, n_folds)
        tr = [index_of[r.patient_id] for r in split.train]
        va = [index_of[r.patient_id] for r in split.val]
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=5000, random_state=seed))
        clf.fit(features[tr], y[tr])
        p = clf.predict(features[va])
        true.append(y[va]); pred.append(p)
        fold_f1.append(f1_score(y[va], p, average="macro"))
    t, p = np.concatenate(true), np.concatenate(pred)
    osteo = config.CLASS_NAMES.index("Osteoporosis")
    return {
        "val_macro_f1": float(f1_score(t, p, average="macro")),
        "val_macro_f1_fold_sd": float(np.std(fold_f1, ddof=1)),
        "val_balanced_accuracy": float(balanced_accuracy_score(t, p)),
        "val_accuracy": float((t == p).mean()),
        "val_osteoporosis_recall": float(((p == osteo) & (t == osteo)).sum() / max(1, (t == osteo).sum())),
    }


def main() -> None:
    data_dir = config.DATA_ARTIFACTS_DIR
    records = [r for r in load_manifest(data_dir / "manifest.csv") if not r.excluded]
    parts, n_folds, seed = load_parts(data_dir / "folds.json")
    index_of = {r.patient_id: i for i, r in enumerate(records)}
    cache = config.ARTIFACTS_DIR / "cache"

    configs = {"plain": config.PREPROCESSING,
               "clahe": dataclasses.replace(config.PREPROCESSING, use_clahe=True, version="prep-v1-clahe")}
    results = {}
    X_by_prep = {}
    for prep_name, prep in configs.items():
        X, y = load_arrays(records, config.DATASET_DIR, prep, config.CLASS_NAMES, cache)
        X_by_prep[prep_name] = X
    majority = np.bincount(y).argmax()
    results["majority_class (always Osteopenia)"] = {
        "val_macro_f1": float(f1_score(y, np.full_like(y, majority), average="macro")),
        "val_balanced_accuracy": 1 / 3, "val_accuracy": float((y == majority).mean()),
        "val_osteoporosis_recall": 0.0, "val_macro_f1_fold_sd": 0.0}
    results["shortcut baseline (size/format/brightness)"] = probe(
        shortcut_features(records), y, index_of, records, parts, n_folds, seed)

    for backbone in ("efficientnet_b0", "resnet50", "densenet121", "vgg16"):
        for prep_name in (("plain", "clahe") if backbone == "efficientnet_b0" else ("plain",)):
            started = time.time()
            features = extract_features(backbone, X_by_prep[prep_name])
            key = f"{backbone} ({prep_name})"
            results[key] = probe(features, y, index_of, records, parts, n_folds, seed)
            results[key]["feature_seconds"] = round(time.time() - started, 1)
            print(f"{key:<44} macroF1 {results[key]['val_macro_f1']:.3f}  "
                  f"balAcc {results[key]['val_balanced_accuracy']:.3f}", flush=True)

    out = config.ARTIFACTS_DIR / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_val.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["# Backbone benchmark — frozen features + logistic regression", "",
             f"Pooled VALIDATION predictions over {n_folds} folds (n = {len(y)}). Test parts not used.", "",
             "| Model | Macro F1 | (fold SD) | Balanced acc. | Accuracy | Osteoporosis recall |",
             "|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['val_macro_f1']:.3f} | {m['val_macro_f1_fold_sd']:.3f} | "
                     f"{m['val_balanced_accuracy']:.3f} | {m['val_accuracy']:.3f} | {m['val_osteoporosis_recall']:.3f} |")
    (out / "benchmark_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
