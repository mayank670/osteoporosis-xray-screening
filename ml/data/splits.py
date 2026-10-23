"""Leakage-safe, reproducible cross-validation folds.

With only ~210 images a single 70/15/15 split would leave ~5 Normal images in
the test set, so we use K-fold cross-validation with a rotating scheme:

    fold k:  test = part k,  validation = part (k+1) mod K,  train = the rest

With K = 5 every image is tested exactly once (60/20/20 per fold). Parts are
built with StratifiedGroupKFold, which keeps class proportions similar in each
part AND never splits a group (patient / near-duplicate cluster) across parts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ml.data.manifest import ImageRecord


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]


def assign_parts(records: list[ImageRecord], n_folds: int, seed: int) -> dict[str, int]:
    """Return {patient_id: part index 0..n_folds-1} for the non-excluded records."""
    usable = [r for r in records if not r.excluded]
    labels = np.array([r.label for r in usable])
    groups = np.array([r.group or r.patient_id for r in usable])
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    parts = {}
    # split() yields (train_idx, test_idx) pairs; we only need each test part.
    for part, (_, test_idx) in enumerate(splitter.split(np.zeros(len(usable)), labels, groups)):
        for i in test_idx:
            parts[usable[i].patient_id] = part
    return parts


def make_fold(records: list[ImageRecord], parts: dict[str, int], fold: int, n_folds: int) -> FoldSplit:
    usable = [r for r in records if not r.excluded]
    val_part = (fold + 1) % n_folds
    return FoldSplit(
        fold=fold,
        train=[r for r in usable if parts[r.patient_id] not in (fold, val_part)],
        val=[r for r in usable if parts[r.patient_id] == val_part],
        test=[r for r in usable if parts[r.patient_id] == fold],
    )


def check_no_leakage(split: FoldSplit) -> None:
    """Raise if any patient, file hash or group appears in more than one subset."""
    for key in ("patient_id", "sha256", "group"):
        seen = [{getattr(r, key) for r in subset} for subset in (split.train, split.val, split.test)]
        if (seen[0] & seen[1]) or (seen[0] & seen[2]) or (seen[1] & seen[2]):
            raise AssertionError(f"Leakage in fold {split.fold}: shared {key}")


def save_parts(parts: dict[str, int], n_folds: int, seed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_folds": n_folds, "seed": seed, "parts": parts}, indent=1), encoding="utf-8")


def load_parts(path: Path) -> tuple[dict[str, int], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["parts"], data["n_folds"], data["seed"]


def split_summary(records: list[ImageRecord], parts: dict[str, int], n_folds: int, class_names) -> list[dict]:
    """Rows of {fold, subset, class counts} for reporting."""
    rows = []
    for fold in range(n_folds):
        split = make_fold(records, parts, fold, n_folds)
        for subset in ("train", "val", "test"):
            items = getattr(split, subset)
            row = {"fold": fold, "subset": subset, "total": len(items)}
            row.update({c: sum(r.label == c for r in items) for c in class_names})
            rows.append(row)
    return rows


def write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
