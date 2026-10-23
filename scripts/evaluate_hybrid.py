"""Compare image-only, clinical-only and hybrid predictions on val or test folds.

    python scripts/evaluate_hybrid.py --run-name effb0_baseline --subset val
    python scripts/evaluate_hybrid.py --run-name effb0_baseline --subset test   # once

Image scores come from each fold's CNN, clinical scores from each fold's
clinical_model.json, and hybrid = fuse_scores(clinical, image, w) with the
weight stored in that file. Test runs are appended to
artifacts/test_evaluations.log.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from evaluate import fmt_ci, write_markdown
from ml.data.clinical import load_clinical_table
from ml.data.loader import load_arrays
from ml.data.manifest import load_manifest
from ml.data.splits import load_parts, make_fold
from ml.evaluation.metrics import compute_metrics
from ml.evaluation.plots import plot_confusion_matrix, plot_roc
from ml.models.clinical import ClinicalModel, fuse_scores, screening_decision

MODES = ("image", "clinical", "hybrid", "hybrid_screening")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--subset", choices=["val", "test"], required=True)
    args = parser.parse_args()

    import keras

    import ml.models  # noqa: F401  registers custom layers

    run_dir = config.ARTIFACTS_DIR / "runs" / args.run_name
    records = load_manifest(run_dir / "manifest.csv")
    parts, n_folds, _ = load_parts(run_dir / "folds.json")
    clinical_table = load_clinical_table(config.DATASET_DIR)

    pooled = {m: ([], []) for m in MODES}
    pooled_pred = {m: [] for m in MODES}
    per_fold = {m: {} for m in MODES}
    for fold in range(n_folds):
        fold_dir = run_dir / f"fold_{fold}"
        subset = getattr(make_fold(records, parts, fold, n_folds), args.subset)
        X, y = load_arrays(subset, config.DATASET_DIR, config.PREPROCESSING, config.CLASS_NAMES,
                           config.ARTIFACTS_DIR / "cache")
        image_scores = keras.models.load_model(fold_dir / "model.keras").predict(X, verbose=0)
        clinical_model = ClinicalModel.load(fold_dir / "clinical_model.json")
        clinical_scores = clinical_model.predict_proba(np.array([clinical_table[r.patient_id] for r in subset]))
        fused = fuse_scores(clinical_scores, image_scores, clinical_model.fusion_image_weight)
        # The screening rule changes only the chosen label, never the scores,
        # so ROC-AUC (a ranking metric) is identical to plain "hybrid".
        chosen = np.array([screening_decision(row, config.CLASS_NAMES, clinical_model.osteoporosis_boost)
                           for row in fused])
        scores = {"image": image_scores, "clinical": clinical_scores, "hybrid": fused, "hybrid_screening": fused}
        preds = {"image": None, "clinical": None, "hybrid": None, "hybrid_screening": chosen}
        for mode in MODES:
            pred = preds[mode] if preds[mode] is not None else scores[mode].argmax(axis=1)
            per_fold[mode][fold] = compute_metrics(y, scores[mode], config.CLASS_NAMES, y_pred=pred)
            pooled[mode][0].append(y)
            pooled[mode][1].append(scores[mode])
            pooled_pred[mode].append(pred)
        print(f"fold {fold}: " + "  ".join(f"{m} F1={per_fold[m][fold]['macro_f1']:.3f}" for m in MODES))

    out_dir = run_dir / f"evaluation_hybrid_{args.subset}"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for mode in MODES:
        y_true, y_score = np.concatenate(pooled[mode][0]), np.concatenate(pooled[mode][1])
        metrics = compute_metrics(y_true, y_score, config.CLASS_NAMES, y_pred=np.concatenate(pooled_pred[mode]))
        summary[mode] = metrics
        write_markdown(out_dir / f"report_{mode}.md", f"{args.run_name} [{mode}]", args.subset,
                       metrics, per_fold[mode], config.CLASS_NAMES)
        plot_confusion_matrix(metrics["confusion_matrix"], config.CLASS_NAMES, out_dir / f"confusion_{mode}.png",
                              f"{mode} — pooled {args.subset} (n={metrics['n']})")
        if mode == "hybrid":
            plot_roc(y_true, y_score, config.CLASS_NAMES, out_dir / "roc_hybrid.png")
        o = metrics["per_class"]["Osteoporosis"]
        print(f"{mode:<9} accuracy {metrics['accuracy']:.3f} {fmt_ci(metrics['accuracy_95ci'])}  "
              f"macroF1 {metrics['macro_f1']:.3f} {fmt_ci(metrics['macro_f1_95ci'])}  "
              f"AUC {metrics['roc_auc_macro_ovr']:.3f}  osteo recall {o['recall']:.3f} {fmt_ci(o['recall_95ci'])}")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if args.subset == "test":
        with (config.ARTIFACTS_DIR / "test_evaluations.log").open("a", encoding="utf-8") as log:
            for mode in MODES:
                log.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{args.run_name}[{mode}]\t"
                          f"accuracy={summary[mode]['accuracy']:.4f}\tmacro_f1={summary[mode]['macro_f1']:.4f}\n")
    print(f"Reports: {out_dir}")


if __name__ == "__main__":
    main()
