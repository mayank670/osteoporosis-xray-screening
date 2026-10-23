"""Build a small, upload-friendly copy of both datasets AND audit them for duplicates.

    python scripts/build_compact_dataset.py --kaggle "D:/Projects/Oestoporosis/OS Collected Data" \
        --out "D:/Projects/Oestoporosis/colab_package"

For every image (Mendeley + Kaggle):
  * SHA-256 of the ORIGINAL file (exact-copy detection)
  * grayscale, longest side <= 512 px, saved as JPEG quality 95 (~50 KB instead of ~600 KB).
    512 px is still more than twice the 224-px model input; grayscale is step 1 of our
    preprocessing anyway, so no information the model uses is thrown away.
  * a 128x128 standardised thumbnail (same method as ml/data/manifest.py), compared against
    every other image AND its flipped / rotated versions (disguised copies)

Outputs in --out:
  images/<source>/<label>/<file>.jpg   the compact images
  images.csv                            one row per image incl. duplicate group
  audit_summary.txt                     the numbers to report
  compact_dataset.zip                   everything above + Mendeley spreadsheet, ready for Colab
"""

import argparse
import collections
import csv
import hashlib
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.manifest import thumbnail_vector
from ml.preprocessing import InvalidImageError, decode_image
from ml.preprocessing.transforms import to_grayscale, to_uint8

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_SIDE = 512
THRESH = 0.985  # same calibrated threshold as ml/data/manifest.py
MENDELEY_LABELS = {"normal": "Normal", "osteopenia": "Osteopenia", "osteoporosis": "Osteoporosis"}
KAGGLE_LABELS = {"Normal": "Normal", "Osteopenia": "Osteopenia", "Osteoporosis": "Osteoporosis"}


def safe_name(name: str) -> str:
    """'Normal 103.JPEG' -> 'Normal_103' (no spaces or odd characters in paths)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")


def collect(root: Path, labels: dict, source: str):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in SUFFIXES and path.parent.name in labels:
            yield source, labels[path.parent.name], path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)

    items = list(collect(config.DATASET_DIR, MENDELEY_LABELS, "mendeley")) + \
        list(collect(args.kaggle, KAGGLE_LABELS, "kaggle"))
    print(f"{len(items)} images to process", flush=True)

    rows, thumbs = [], []
    for n, (source, label, path) in enumerate(items, 1):
        data = path.read_bytes()
        row = {"source": source, "label": label, "original_path": str(path.relative_to(
            config.DATASET_DIR if source == "mendeley" else args.kaggle)).replace("\\", "/"),
            "sha256": hashlib.sha256(data).hexdigest(), "error": ""}
        try:
            image = decode_image(data)
        except InvalidImageError as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        row.update(orig_height=image.shape[0], orig_width=image.shape[1], orig_dtype=str(image.dtype),
                   orig_channels=1 if image.ndim == 2 else image.shape[2])
        gray = to_uint8(to_grayscale(image))
        del image  # free the full-resolution array immediately
        scale = min(1.0, MAX_SIDE / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        rel = f"images/{source}/{label}/{safe_name(path.name)}.jpg"
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        target.write_bytes(buf.tobytes())
        row["compact_path"] = rel
        rows.append(row)
        thumbs.append((len(rows) - 1, thumbnail_vector(gray).reshape(128, 128).astype(np.float32)))
        if n % 200 == 0:
            print(f"  {n}/{len(items)}", flush=True)

    # ---- disguised copies: compare originals with flipped/rotated thumbnails ----
    idx = [i for i, _ in thumbs]
    T = np.stack([t for _, t in thumbs])  # (N, 128, 128)
    base = T.reshape(len(T), -1)
    variants = {"orig": T, "hflip": T[:, :, ::-1], "vflip": T[:, ::-1, :], "rot180": T[:, ::-1, ::-1],
                "rot90": np.rot90(T, 1, axes=(1, 2)), "rot270": np.rot90(T, 3, axes=(1, 2))}
    best = np.full((len(T), len(T)), -1.0, np.float32)
    kind = np.zeros((len(T), len(T)), np.int8)
    for k, (name, V) in enumerate(variants.items()):
        C = base @ np.ascontiguousarray(V).reshape(len(T), -1).T
        better = C > best
        best[better], kind[better] = C[better], k
    np.fill_diagonal(best, -1)
    sym = np.maximum(best, best.T)

    pairs = set(zip(*np.nonzero(np.triu(sym, 1) >= THRESH)))
    by_sha = collections.defaultdict(list)
    for j, i in enumerate(idx):
        by_sha[rows[i]["sha256"]].append(j)
    for members in by_sha.values():
        pairs |= {(a, b) for a in members for b in members if a < b}
    parent = list(range(len(T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    kinds = list(variants)
    for j, i in enumerate(idx):
        m = int(sym[j].argmax())
        rows[i].update(group=f"G{find(j):05d}", max_corr=round(float(sym[j, m]), 4),
                       best_match=f"{rows[idx[m]]['source']}:{rows[idx[m]]['original_path']}",
                       match_transform=kinds[kind[j, m] if best[j, m] >= best[m, j] else kind[m, j]])

    # ---- summary ----
    lines = []
    ok_rows = [r for r in rows if not r["error"]]
    for source in ("mendeley", "kaggle"):
        rs = [r for r in ok_rows if r["source"] == source]
        lines.append(f"{source.upper()}: {len(rs)} images {dict(collections.Counter(r['label'] for r in rs))}")
        lines.append(f"  original sizes (most common): {collections.Counter((r['orig_height'], r['orig_width']) for r in rs).most_common(4)}")
    lines.append(f"unreadable files: {sum(bool(r['error']) for r in rows)}")
    groups = collections.defaultdict(list)
    for r in ok_rows:
        groups[r["group"]].append(r)
    m_sha = {r["sha256"] for r in ok_rows if r["source"] == "mendeley"}
    k_rows = [r for r in ok_rows if r["source"] == "kaggle"]
    lines.append(f"kaggle files byte-identical to a Mendeley file: {sum(r['sha256'] in m_sha for r in k_rows)}")
    lines.append(f"matched pairs by transform: {dict(collections.Counter(kinds[kind[a, b] if best[a, b] >= best[b, a] else kind[b, a]] for a, b in pairs))}")
    summary, new_by_class, new_size = collections.Counter(), collections.Counter(), collections.Counter()
    for members in groups.values():
        ks = [r for r in members if r["source"] == "kaggle"]
        if not ks:
            continue
        labels = {r["label"] for r in members}
        if len(labels) > 1:
            summary["kaggle groups with CONFLICTING labels"] += 1
            summary["  kaggle files in them"] += len(ks)
        elif any(r["source"] == "mendeley" for r in members):
            summary["kaggle groups that are Mendeley images"] += 1
            summary["  kaggle files in them"] += 0
            summary["kaggle files that copy Mendeley"] += len(ks)
        else:
            summary["genuinely NEW kaggle X-rays (groups)"] += 1
            new_by_class[ks[0]["label"]] += 1
            new_size[(ks[0]["label"], "large>=3000px" if max(ks[0]["orig_height"], ks[0]["orig_width"]) >= 3000 else "small<3000px")] += 1
    lines += [f"{k}: {v}" for k, v in summary.items() if v]
    lines.append(f"GENUINELY NEW per class: {dict(new_by_class)}  total {sum(new_by_class.values())}")
    lines.append(f"new images by class and original size (scanner-shortcut check): {dict(sorted(new_size.items()))}")
    text = "\n".join(lines)
    (out / "audit_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)

    fields = ["source", "label", "original_path", "compact_path", "sha256", "group", "max_corr", "best_match",
              "match_transform", "orig_height", "orig_width", "orig_dtype", "orig_channels", "error"]
    with (out / "images.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    sheet = next(config.DATASET_DIR.glob("*.xlsx"))
    with zipfile.ZipFile(out / "compact_dataset.zip", "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
        for r in ok_rows:
            z.write(out / r["compact_path"], r["compact_path"])
        z.write(out / "images.csv", "images.csv")
        z.write(out / "audit_summary.txt", "audit_summary.txt")
        z.write(sheet, f"mendeley_clinical/{sheet.name}")
    size_mb = (out / "compact_dataset.zip").stat().st_size / 1e6
    print(f"\nPackage: {out / 'compact_dataset.zip'}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
