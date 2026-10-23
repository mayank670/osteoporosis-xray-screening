"""Train the classifier on every cross-validation fold (validation data only).

Run:  python scripts/prepare_dataset.py      (once)
      python scripts/train.py --run-name effb0_baseline
      python scripts/train.py --backbone resnet50 --run-name resnet50 --folds 0

Output: artifacts/runs/<run-name>/fold_<k>/ with model.keras, class_map.json,
preprocessing.json, metadata.json, history.json, training_log.csv.
The test subsets are NOT used here — see scripts/evaluate.py.
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.loader import load_arrays
from ml.data.manifest import load_manifest
from ml.data.splits import check_no_leakage, load_parts, make_fold
from ml.training.artifacts import write_model_metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backbone", default=config.TRAINING.backbone)
    parser.add_argument("--folds", type=int, nargs="*", help="default: all folds")
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--clahe", action="store_true", help="enable CLAHE (experiment)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"backbone": args.backbone}
    if args.head_epochs is not None:
        overrides["head_epochs"] = args.head_epochs
    if args.finetune_epochs is not None:
        overrides["finetune_epochs"] = args.finetune_epochs
    train_cfg = dataclasses.replace(config.TRAINING, **overrides)
    prep_cfg = config.PREPROCESSING
    if args.clahe:
        prep_cfg = dataclasses.replace(prep_cfg, use_clahe=True, version=prep_cfg.version + "-clahe")

    data_dir = config.DATA_ARTIFACTS_DIR
    records = load_manifest(data_dir / "manifest.csv")
    parts, n_folds, seed = load_parts(data_dir / "folds.json")
    if n_folds != train_cfg.n_folds or seed != train_cfg.seed:
        sys.exit("folds.json does not match TrainingConfig; re-run scripts/prepare_dataset.py")

    run_dir = config.ARTIFACTS_DIR / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "manifest.csv", run_dir / "manifest.csv")
    shutil.copy(data_dir / "folds.json", run_dir / "folds.json")
    (run_dir / "run_config.json").write_text(json.dumps(
        {"training": train_cfg.to_dict(), "preprocessing": prep_cfg.to_dict()}, indent=1), encoding="utf-8")

    from ml.training.trainer import train_fold

    for fold in args.folds if args.folds else range(n_folds):
        split = make_fold(records, parts, fold, n_folds)
        check_no_leakage(split)
        X_train, y_train = load_arrays(split.train, config.DATASET_DIR, prep_cfg, config.CLASS_NAMES,
                                       config.ARTIFACTS_DIR / "cache")
        X_val, y_val = load_arrays(split.val, config.DATASET_DIR, prep_cfg, config.CLASS_NAMES,
                                   config.ARTIFACTS_DIR / "cache")
        print(f"\n=== {args.run_name} fold {fold}: train {len(y_train)}  val {len(y_val)} ===", flush=True)
        started = time.time()
        fold_dir = run_dir / f"fold_{fold}"
        _, history = train_fold(X_train, y_train, X_val, y_val, train_cfg, prep_cfg, config.CLASS_NAMES, fold_dir)
        write_model_metadata(
            fold_dir, config.CLASS_NAMES, prep_cfg.to_dict(), train_cfg.to_dict(),
            model_version=f"{args.run_name}-fold{fold}",
            extra={"fold": fold, "train_size": len(y_train), "val_size": len(y_val),
                   "best_val_loss": history["best_val_loss"], "train_seconds": round(time.time() - started)},
        )
        print(f"fold {fold}: best val_loss {history['best_val_loss']:.4f} "
              f"({len(history['loss'])} epochs, {time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
