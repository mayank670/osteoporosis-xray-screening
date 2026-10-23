"""Build a cleaned, auditable list of dataset images (the "manifest").

A manifest row records where an image is, its label, which participant it
belongs to, and a SHA-256 content hash (identical bytes -> identical hash).
Every later step (splitting, training, evaluation) reads the manifest instead
of re-scanning folders, so the exact images used are written down and checkable.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ml.preprocessing import InvalidImageError, decode_image
from ml.preprocessing.transforms import letterbox_resize, to_grayscale, to_uint8

# Dataset folder name -> class name used by the model (config.CLASS_NAMES).
FOLDER_TO_CLASS = {"normal": "Normal", "osteopenia": "Osteopenia", "osteoporosis": "Osteoporosis"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# Pearson correlation of 128x128 thumbnails above which two images are treated
# as near-copies. On this dataset the most similar pair of *different* patients
# scores 0.976 (same scanner and positioning). Measured: half-size + JPEG-70
# copies score 1.000, but a copy cropped by 20 px can score as low as 0.915, so
# cropped copies are NOT reliably caught (documented limitation).
NEAR_DUPLICATE_MIN_CORRELATION = 0.985
THUMBNAIL_SIZE = 128


@dataclass
class ImageRecord:
    path: str  # relative to the dataset root, forward slashes
    label: str
    patient_id: str
    sha256: str
    group: str = ""  # images that must stay in the same split share a group
    excluded: bool = False
    reason: str = ""
    near_duplicates: list[str] = field(default_factory=list)


def thumbnail_vector(image: np.ndarray, size: int = THUMBNAIL_SIZE) -> np.ndarray:
    """Small standardised copy of the image, flattened to a unit-length vector.

    Subtracting the mean and dividing by the standard deviation removes
    brightness/contrast differences; dividing by sqrt(n) makes the dot product
    of two such vectors equal to their Pearson correlation (-1..1).
    """
    gray = letterbox_resize(to_uint8(to_grayscale(image)), (size, size)).astype(np.float32).ravel()
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray / np.sqrt(gray.size)


def scan_dataset(root: Path) -> tuple[list[ImageRecord], dict[str, np.ndarray]]:
    """Read every image under root/<class folder>/.

    Returns the records and a {patient_id: thumbnail vector} dict for the
    near-duplicate check (kept separately so the manifest stays small).
    """
    records, thumbnails = [], {}
    for folder, label in FOLDER_TO_CLASS.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            data = path.read_bytes()
            try:
                image = decode_image(data)
            except InvalidImageError as exc:
                records.append(ImageRecord(path.relative_to(root).as_posix(), label, path.stem.upper(),
                                           hashlib.sha256(data).hexdigest(), excluded=True,
                                           reason=f"unreadable: {exc}"))
                continue
            records.append(ImageRecord(
                path=path.relative_to(root).as_posix(),
                label=label,
                patient_id=path.stem.upper(),  # e.g. "OP12" — matches the spreadsheet
                sha256=hashlib.sha256(data).hexdigest(),
            ))
            thumbnails[path.stem.upper()] = thumbnail_vector(image)
    return records, thumbnails


def remove_exact_duplicates(records: list[ImageRecord]) -> None:
    """Mark byte-identical files. Conflicting labels: exclude all. Same label: keep one."""
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if not record.excluded:
            by_hash[record.sha256].append(record)
    for group in by_hash.values():
        if len(group) < 2:
            continue
        names = ", ".join(r.patient_id for r in group)
        if len({r.label for r in group}) > 1:
            for record in group:
                record.excluded, record.reason = True, f"identical file with conflicting labels ({names})"
        else:
            for record in sorted(group, key=_id_order)[1:]:
                record.excluded, record.reason = True, f"exact duplicate of {sorted(group, key=_id_order)[0].patient_id}"


def assign_groups(
    records: list[ImageRecord],
    thumbnails: dict[str, np.ndarray],
    min_correlation: float = NEAR_DUPLICATE_MIN_CORRELATION,
) -> None:
    """Union near-duplicate images into shared groups (union-find).

    If A~B and B~C then A, B and C share one group, so no pair of look-alike
    images can be split across train and test.
    """
    kept = [r for r in records if not r.excluded]
    parent = {r.patient_id: r.patient_id for r in kept}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving keeps trees shallow
            x = parent[x]
        return x

    if kept:
        vectors = np.stack([thumbnails[r.patient_id] for r in kept])
        correlation = vectors @ vectors.T  # (N, N) matrix of all pairwise correlations
        rows, cols = np.nonzero(np.triu(correlation, k=1) >= min_correlation)
        for i, j in zip(rows, cols):
            a, b = kept[i], kept[j]
            a.near_duplicates.append(b.patient_id)
            b.near_duplicates.append(a.patient_id)
            parent[find(a.patient_id)] = find(b.patient_id)
    for record in kept:
        record.group = find(record.patient_id)


def build_manifest(root: Path) -> list[ImageRecord]:
    records, thumbnails = scan_dataset(root)
    remove_exact_duplicates(records)
    assign_groups(records, thumbnails)
    return records


def _id_order(record: ImageRecord) -> tuple[str, int]:
    """Sort 'OP2' before 'OP10' (natural order)."""
    prefix = record.patient_id.rstrip("0123456789")
    number = record.patient_id[len(prefix):]
    return prefix, int(number) if number else -1


CSV_FIELDS = ["path", "label", "patient_id", "group", "sha256", "excluded", "reason", "near_duplicates"]


def save_manifest(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["near_duplicates"] = ";".join(record.near_duplicates)
            writer.writerow(row)


def load_manifest(path: Path) -> list[ImageRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            ImageRecord(
                path=row["path"], label=row["label"], patient_id=row["patient_id"],
                sha256=row["sha256"], group=row["group"],
                excluded=row["excluded"] == "True", reason=row["reason"],
                near_duplicates=[x for x in row["near_duplicates"].split(";") if x],
            )
            for row in csv.DictReader(handle)
        ]
