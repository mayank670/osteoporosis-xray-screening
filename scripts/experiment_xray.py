"""X-ray-ONLY improvement experiments (validation folds only). Designed for a GPU (Colab T4).

    python scripts/experiment_xray.py                       # all variants, both tasks
    python scripts/experiment_xray.py --variants knees224 --tasks binary --folds 0 --quick

Variants (how each image becomes model input):
  whole224   current pipeline: whole picture letterboxed to 224 (baseline)
  crop224    black borders cropped away first, then 224
  knees224   two-knee images split into single knees (right knee mirrored), each 224
  knees384   same, at 384 px (more bone texture)

Tasks:
  3class     Normal / Osteopenia / Osteoporosis
  binary     Normal vs Osteoporosis (Osteopenia patients left out)

Leakage rules: folds are by PATIENT (from artifacts/data/folds.json); both knees of a patient
always share a fold; a patient's score = mean of its knees' scores. Test parts are never used.
Output: artifacts/experiments/xray/summary.{md,json}  (models are deleted to save space)
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.manifest import load_manifest
from ml.data.splits import check_no_leakage, load_parts, make_fold
from ml.evaluation.metrics import compute_metrics
from ml.preprocessing import read_image_file
from ml.preprocessing.knee import crop_to_content, split_knees
from ml.preprocessing.transforms import letterbox_resize, to_grayscale, to_model_input, to_uint8

VARIANTS = {"whole224": ("whole", 224), "crop224": ("crop", 224), "knees224": ("knees", 224), "knees384": ("knees", 384)}
TASKS = {"3class": ("Normal", "Osteopenia", "Osteoporosis"), "binary": ("Normal", "Osteoporosis")}


def samples_for(records, mode: str, size: int):
    """Return X (n, size, size, 3) float32 0..255, and the patient id of every sample."""
    X, owners = [], []
    for r in records:
        gray = to_uint8(to_grayscale(read_image_file(config.DATASET_DIR / r.path)))
        if mode == "whole":
            parts = [gray]
        elif mode == "crop":
            parts = [crop_to_content(gray)]
        else:
            parts = split_knees(gray)
        for part in parts:
            X.append(to_model_input(letterbox_resize(part, (size, size))))
            owners.append(r.patient_id)
    return np.stack(X), np.array(owners)


def patient_scores(scores: np.ndarray, owners: np.ndarray, patients: list[str]) -> np.ndarray:
    """Average the knee-level scores of each patient (order = `patients`)."""
    return np.stack([scores[owners == p].mean(axis=0) for p in patients])


def run(variant: str, task: str, records, parts, n_folds, folds, quick: bool, out_root: Path) -> dict:
    from ml.training.trainer import train_fold

    mode, size = VARIANTS[variant]
    classes = TASKS[task]
    usable = [r for r in records if not r.excluded and r.label in classes]
    X_all, owners = samples_for(usable, mode, size)
    label_of = {r.patient_id: classes.index(r.label) for r in usable}
    y_all = np.array([label_of[o] for o in owners], dtype=np.int32)
    print(f"\n##### {variant} / {task}: {len(usable)} patients -> {len(X_all)} samples at {size}px", flush=True)

    cfg = dataclasses.replace(config.TRAINING, deterministic_ops=False,
                              head_epochs=3 if quick else config.TRAINING.head_epochs,
                              finetune_epochs=1 if quick else config.TRAINING.finetune_epochs)
    prep = dataclasses.replace(config.PREPROCESSING, image_size=(size, size), version=f"exp-{variant}")
    all_true, all_scores, fold_rows = [], [], []
    for fold in folds:
        split = make_fold(records, parts, fold, n_folds)
        check_no_leakage(split)
        train_ids = {r.patient_id for r in split.train if r.label in classes}
        val_patients = [r.patient_id for r in split.val if r.label in classes]
        tr = np.isin(owners, list(train_ids))
        va = np.isin(owners, val_patients)
        started = time.time()
        fold_dir = out_root / f"{variant}_{task}" / f"fold_{fold}"
        model, history = train_fold(X_all[tr], y_all[tr], X_all[va], y_all[va], cfg, prep, classes, fold_dir)
        knee_scores = model.predict(X_all[va], verbose=0)
        scores = patient_scores(knee_scores, owners[va], val_patients)
        y_true = np.array([label_of[p] for p in val_patients])
        m = compute_metrics(y_true, scores, classes)
        fold_rows.append({"fold": fold, "macro_f1": m["macro_f1"], "auc": m["roc_auc_macro_ovr"],
                          "epochs": len(history["loss"]), "seconds": round(time.time() - started)})
        print(f"  fold {fold}: macroF1 {m['macro_f1']:.3f}  AUC {m['roc_auc_macro_ovr']}  "
              f"({len(history['loss'])} epochs, {time.time() - started:.0f}s)", flush=True)
        all_true.append(y_true)
        all_scores.append(scores)
        shutil.rmtree(fold_dir, ignore_errors=True)  # models not needed; saves disk space
        del model
    pooled = compute_metrics(np.concatenate(all_true), np.concatenate(all_scores), classes)
    result = {"variant": variant, "task": task, "n_patients": pooled["n"], "n_samples": int(len(X_all)),
              "macro_f1": pooled["macro_f1"], "macro_f1_95ci": pooled["macro_f1_95ci"],
              "balanced_accuracy": float(np.mean([pooled["per_class"][c]["recall"] for c in classes])),
              "accuracy": pooled["accuracy"], "auc": pooled["roc_auc_macro_ovr"],
              "osteoporosis_recall": pooled["per_class"]["Osteoporosis"]["recall"],
              "normal_recall": pooled["per_class"]["Normal"]["recall"],
              "confusion_matrix": pooled["confusion_matrix"], "folds": fold_rows}
    print(f"  POOLED VAL: macroF1 {result['macro_f1']:.3f}  balAcc {result['balanced_accuracy']:.3f}  "
          f"AUC {result['auc']}  osteo recall {result['osteoporosis_recall']:.3f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--quick", action="store_true", help="3+1 epochs: checks the code runs, results meaningless")
    args = parser.parse_args()

    records = load_manifest(config.DATA_ARTIFACTS_DIR / "manifest.csv")
    parts, n_folds, _ = load_parts(config.DATA_ARTIFACTS_DIR / "folds.json")
    folds = args.folds if args.folds else list(range(n_folds))
    out_root = config.ARTIFACTS_DIR / "experiments" / "xray"
    out_root.mkdir(parents=True, exist_ok=True)

    import keras

    backend = keras.backend.backend()
    if backend == "jax":
        import jax

        devices = jax.devices()
    else:
        import tensorflow as tf

        devices = tf.config.list_physical_devices("GPU") or ["CPU only - slow"]
    print(f"Keras {keras.__version__} | backend: {backend} | devices: {devices}", flush=True)
    results = []
    for task in args.tasks:
        for variant in args.variants:
            results.append(run(variant, task, records, parts, n_folds, folds, args.quick, out_root))
            (out_root / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    lines = ["# X-ray-only experiments - VALIDATION folds, patient level", "",
             f"Folds: {folds}{'  (QUICK MODE - not meaningful)' if args.quick else ''}", "",
             "| Task | Variant | Patients | Macro F1 (95% CI) | Balanced acc. | ROC-AUC | Osteoporosis recall | Normal recall |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "-"
        lines.append(f"| {r['task']} | {r['variant']} | {r['n_patients']} | {r['macro_f1']:.3f} "
                     f"({r['macro_f1_95ci'][0]:.2f}-{r['macro_f1_95ci'][1]:.2f}) | {r['balanced_accuracy']:.3f} | {auc} | "
                     f"{r['osteoporosis_recall']:.3f} | {r['normal_recall']:.3f} |")
    text = "\n".join(lines) + "\n"
    (out_root / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
