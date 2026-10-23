"""Evaluate a trained run on its validation or held-out test subsets.

    python scripts/evaluate.py --run-name effb0_baseline --subset val    # for model comparisons
    python scripts/evaluate.py --run-name effb0_baseline --subset test   # ONCE, after decisions

Each fold's model predicts only its own subset, so with --subset test every
usable image is predicted exactly once by a model that never saw it. These
"out-of-fold" predictions are pooled into one set of metrics (n = 210).

Every --subset test run is appended to artifacts/test_evaluations.log, so any
repeated use of the test set is visible in the project record.
"""

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.loader import load_arrays
from ml.data.manifest import load_manifest
from ml.data.splits import load_parts, make_fold
from ml.evaluation.metrics import compute_metrics
from ml.evaluation.plots import plot_confusion_matrix, plot_history, plot_roc


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_markdown(path: Path, run: str, subset: str, pooled: dict, per_fold: dict, class_names) -> None:
    lines = [f"# Evaluation — `{run}` — {subset} subsets", "",
             f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Labels are QUS T-score categories.", "",
             f"## Pooled out-of-fold results (n = {pooled['n']})", "",
             "| Metric | Value | 95% CI |", "|---|---|---|",
             f"| Accuracy | {pooled['accuracy']:.3f} | {fmt_ci(pooled['accuracy_95ci'])} (Wilson) |",
             f"| Macro F1 | {pooled['macro_f1']:.3f} | {fmt_ci(pooled['macro_f1_95ci'])} (bootstrap) |"]
    if pooled.get("roc_auc_macro_ovr") is not None:
        lines.append(f"| ROC-AUC (macro, one-vs-rest) | {pooled['roc_auc_macro_ovr']:.3f} | — |")
    lines += ["", "| Class | Precision | Recall | Recall 95% CI | F1 | Support |", "|---|---|---|---|---|---|"]
    for name in class_names:
        c = pooled["per_class"][name]
        lines.append(f"| {name} | {c['precision']:.3f} | {c['recall']:.3f} | {fmt_ci(c['recall_95ci'])} "
                     f"| {c['f1']:.3f} | {c['support']} |")
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", "", "```",
              "              " + " ".join(f"{n[:12]:>12}" for n in class_names)]
    for name, row in zip(class_names, pooled["confusion_matrix"]):
        lines.append(f"{name[:12]:>12}  " + " ".join(f"{v:>12}" for v in row))
    lines += ["```", "", "## Per fold", "", "| Fold | n | Accuracy | Macro F1 | Osteoporosis recall |",
              "|---|---|---|---|---|"]
    for fold, m in per_fold.items():
        lines.append(f"| {fold} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} "
                     f"| {m['per_class']['Osteoporosis']['recall']:.3f} |")
    accs = [m["accuracy"] for m in per_fold.values()]
    f1s = [m["macro_f1"] for m in per_fold.values()]
    lines += [f"| mean ± sd | | {np.mean(accs):.3f} ± {np.std(accs, ddof=1) if len(accs) > 1 else 0:.3f} "
              f"| {np.mean(f1s):.3f} ± {np.std(f1s, ddof=1) if len(f1s) > 1 else 0:.3f} | |",
              "", "## Classification report (pooled)", "", "```", pooled["classification_report"], "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--subset", choices=["val", "test"], required=True)
    args = parser.parse_args()

    import keras

    import ml.models  # noqa: F401  registers custom layers before loading

    run_dir = config.ARTIFACTS_DIR / "runs" / args.run_name
    records = load_manifest(run_dir / "manifest.csv")
    parts, n_folds, _ = load_parts(run_dir / "folds.json")
    out_dir = run_dir / f"evaluation_{args.subset}"
    out_dir.mkdir(exist_ok=True)

    all_true, all_score, per_fold, histories, rows = [], [], {}, {}, []
    for fold in range(n_folds):
        fold_dir = run_dir / f"fold_{fold}"
        if not (fold_dir / "model.keras").exists():
            print(f"fold {fold}: no model, skipped")
            continue
        class_names = tuple(json.loads((fold_dir / "class_map.json").read_text())[str(i)] for i in range(3))
        if class_names != config.CLASS_NAMES:
            sys.exit(f"class_map.json mismatch in {fold_dir}")
        prep = config.PreprocessingConfig(**{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in json.loads((fold_dir / "preprocessing.json").read_text()).items()})
        subset = getattr(make_fold(records, parts, fold, n_folds), args.subset)
        X, y = load_arrays(subset, config.DATASET_DIR, prep, class_names, config.ARTIFACTS_DIR / "cache")
        scores = keras.models.load_model(fold_dir / "model.keras").predict(X, verbose=0)
        if not np.isfinite(scores).all():
            sys.exit(f"non-finite scores from fold {fold}")
        per_fold[fold] = compute_metrics(y, scores, class_names)
        histories[fold] = json.loads((fold_dir / "history.json").read_text())
        all_true.append(y)
        all_score.append(scores)
        for record, label, score in zip(subset, y, scores):
            rows.append({"fold": fold, "patient_id": record.patient_id, "true": class_names[label],
                         "predicted": class_names[int(score.argmax())],
                         **{f"score_{c}": f"{s:.4f}" for c, s in zip(class_names, score)}})
        print(f"fold {fold}: n={len(y)} accuracy={per_fold[fold]['accuracy']:.3f} "
              f"macroF1={per_fold[fold]['macro_f1']:.3f}")

    if not per_fold:
        sys.exit("No trained folds found.")
    y_true, y_score = np.concatenate(all_true), np.concatenate(all_score)
    pooled = compute_metrics(y_true, y_score, config.CLASS_NAMES)

    (out_dir / "metrics.json").write_text(json.dumps(
        {"pooled": pooled, "per_fold": {str(k): v for k, v in per_fold.items()}}, indent=1), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(out_dir / "report.md", args.run_name, args.subset, pooled, per_fold, config.CLASS_NAMES)
    plot_confusion_matrix(pooled["confusion_matrix"], config.CLASS_NAMES, out_dir / "confusion_matrix.png",
                          f"{args.run_name} — pooled {args.subset} (n={pooled['n']})")
    plot_roc(y_true, y_score, config.CLASS_NAMES, out_dir / "roc.png")
    plot_history(histories, out_dir / "training_curves.png")

    if args.subset == "test":
        with (config.ARTIFACTS_DIR / "test_evaluations.log").open("a", encoding="utf-8") as log:
            log.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{args.run_name}\t"
                      f"accuracy={pooled['accuracy']:.4f}\tmacro_f1={pooled['macro_f1']:.4f}\n")

    print(f"\nPooled {args.subset}: accuracy {pooled['accuracy']:.3f} {fmt_ci(pooled['accuracy_95ci'])}, "
          f"macro F1 {pooled['macro_f1']:.3f} {fmt_ci(pooled['macro_f1_95ci'])}")
    print(f"Osteoporosis recall {pooled['per_class']['Osteoporosis']['recall']:.3f} "
          f"{fmt_ci(pooled['per_class']['Osteoporosis']['recall_95ci'])}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
