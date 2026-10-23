"""Train the clinical risk-factor model for every fold of an existing run.

    python scripts/train_clinical.py --run-name effb0_baseline

Each fold's model is fitted on that fold's TRAIN part only (the same images
the fold's CNN was trained on) and saved as fold_<k>/clinical_model.json.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.clinical import load_clinical_table
from ml.data.manifest import load_manifest
from ml.data.splits import check_no_leakage, load_parts, make_fold
from ml.models.clinical import DEFAULT_C, DEFAULT_IMAGE_WEIGHT, DEFAULT_OSTEOPOROSIS_BOOST, ClinicalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    run_dir = config.ARTIFACTS_DIR / "runs" / args.run_name
    records = load_manifest(run_dir / "manifest.csv")
    parts, n_folds, seed = load_parts(run_dir / "folds.json")
    clinical = load_clinical_table(config.DATASET_DIR)

    for fold in range(n_folds):
        split = make_fold(records, parts, fold, n_folds)
        check_no_leakage(split)
        X = np.array([clinical[r.patient_id] for r in split.train])
        y = np.array([config.CLASS_NAMES.index(r.label) for r in split.train])
        model = ClinicalModel.fit(X, y, config.CLASS_NAMES, C=DEFAULT_C, seed=seed,
                                  fusion_image_weight=DEFAULT_IMAGE_WEIGHT,
                                  osteoporosis_boost=DEFAULT_OSTEOPOROSIS_BOOST,
                                  note=f"{args.run_name} fold {fold} train part, n={len(y)}")
        model.save(run_dir / f"fold_{fold}" / "clinical_model.json")
        print(f"fold {fold}: clinical model trained on {len(y)} patients")


if __name__ == "__main__":
    main()
