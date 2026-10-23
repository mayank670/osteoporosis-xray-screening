"""Save a figure showing each preprocessing step for one image per class.

Run:  python scripts/visualize_preprocessing.py
Output: artifacts/figures/preprocessing_steps.png
"""

import dataclasses
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no GUI window needed
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.preprocessing import preprocess_image, read_image_file
from ml.preprocessing.transforms import to_grayscale, to_uint8


def describe(name: str, array) -> str:
    return f"{name}\n{array.shape} {array.dtype} [{array.min():.0f}, {array.max():.0f}]"


def main() -> None:
    clahe_cfg = dataclasses.replace(config.PREPROCESSING, use_clahe=True)
    folders = sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(folders), 4, figsize=(14, 3.8 * len(folders)))
    for row, folder in zip(axes, folders):
        path = sorted(folder.glob("*.jp*g"))[0]
        raw = read_image_file(path)
        steps = [
            (f"{folder.name}/{path.name} decoded", raw),
            ("grayscale uint8", to_uint8(to_grayscale(raw))),
            ("model input (ch 0)", preprocess_image(raw)[..., 0]),
            ("model input + CLAHE", preprocess_image(raw, clahe_cfg)[..., 0]),
        ]
        for ax, (title, image) in zip(row, steps):
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_title(describe(title, image), fontsize=8)
            ax.axis("off")
    out = config.ARTIFACTS_DIR / "figures" / "preprocessing_steps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
