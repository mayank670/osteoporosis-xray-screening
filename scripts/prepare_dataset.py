"""Build the cleaned manifest and the cross-validation folds.

Run:  python scripts/prepare_dataset.py
Writes to artifacts/data/:
    manifest.csv        every image, with exclusion reasons and groups
    folds.json          {patient_id: part} — the fixed, seeded fold assignment
    split_summary.csv   class counts per fold and subset
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.manifest import build_manifest, save_manifest
from ml.data.splits import (
    assign_parts, check_no_leakage, make_fold, save_parts, split_summary, write_summary_csv,
)


def main() -> None:
    out = config.DATA_ARTIFACTS_DIR
    records = build_manifest(config.DATASET_DIR)
    save_manifest(records, out / "manifest.csv")

    usable = [r for r in records if not r.excluded]
    print(f"Images found: {len(records)}   usable: {len(usable)}   excluded: {len(records) - len(usable)}")
    print("Usable per class:", dict(collections.Counter(r.label for r in usable)))
    for reason, n in collections.Counter(r.reason.split(" (")[0] for r in records if r.excluded).items():
        print(f"  excluded - {reason}: {n}")
    near = [r for r in usable if r.near_duplicates]
    print(f"Images with a near-duplicate (grouped together): {len(near)}")

    cfg = config.TRAINING
    parts = assign_parts(records, cfg.n_folds, cfg.seed)
    save_parts(parts, cfg.n_folds, cfg.seed, out / "folds.json")
    for fold in range(cfg.n_folds):
        check_no_leakage(make_fold(records, parts, fold, cfg.n_folds))
    print(f"\n{cfg.n_folds} folds created (seed {cfg.seed}); leakage check passed for every fold.")

    rows = split_summary(records, parts, cfg.n_folds, config.CLASS_NAMES)
    write_summary_csv(rows, out / "split_summary.csv")
    print(f"{'fold':>4} {'subset':<6} {'total':>5} " + " ".join(f"{c:>12}" for c in config.CLASS_NAMES))
    for row in rows:
        print(f"{row['fold']:>4} {row['subset']:<6} {row['total']:>5} "
              + " ".join(f"{row[c]:>12}" for c in config.CLASS_NAMES))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
