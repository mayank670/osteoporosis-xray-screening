"""Knee-specific geometry: crop away empty background, split two-knee images.

Many dataset images show BOTH knees side by side (bilateral view). The model
sees each knee at only ~80 px wide after resizing the whole picture to 224.
Splitting gives one image per knee at twice the detail. Both knees come from
the same patient, so they always stay in the same data split (no leakage),
and the patient's prediction is the average of the two knees' scores.

Both functions work on uint8 grayscale (H, W) arrays and are deterministic.
"""

from __future__ import annotations

import numpy as np

BACKGROUND_LEVEL = 25  # pixels darker than this count as empty background (0..255)


def content_box(gray: np.ndarray, level: int = BACKGROUND_LEVEL, margin: float = 0.02) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the region containing anatomy.

    A row/column counts as content if more than 2% of its pixels are brighter
    than `level`. Using a fraction (not "any pixel") ignores specks and text markers.
    """
    mask = gray > level
    rows = np.nonzero(mask.mean(axis=1) > 0.02)[0]
    cols = np.nonzero(mask.mean(axis=0) > 0.02)[0]
    h, w = gray.shape
    if len(rows) == 0 or len(cols) == 0:
        return 0, h, 0, w  # blank image: keep everything
    pad_r, pad_c = int(margin * h), int(margin * w)
    return (max(0, rows[0] - pad_r), min(h, rows[-1] + 1 + pad_r),
            max(0, cols[0] - pad_c), min(w, cols[-1] + 1 + pad_c))


def crop_to_content(gray: np.ndarray, level: int = BACKGROUND_LEVEL) -> np.ndarray:
    top, bottom, left, right = content_box(gray, level)
    return gray[top:bottom, left:right]


def column_profile(gray: np.ndarray) -> np.ndarray:
    """Mean brightness of every column, smoothed over ~5% of the width (moving average)."""
    profile = gray.astype(np.float32).mean(axis=0)
    k = max(3, len(profile) // 20)
    return np.convolve(profile, np.ones(k) / k, mode="same")


def valley_position(profile: np.ndarray) -> int:
    """Column of the darkest point in the middle third: the candidate gap between knees."""
    lo, hi = len(profile) // 3, 2 * len(profile) // 3
    return lo + int(np.argmin(profile[lo:hi]))


def gap_ratio(gray: np.ndarray) -> float:
    """Darkest middle column divided by the DIMMER of the two brightest peaks around it.

    Two knees: dark valley between two bright bone peaks       -> ratio well below 1.
    One knee, centred: the middle third contains bone           -> ratio near 1.
    One knee, off-centre: one side holds only dark background,
    so its "peak" is dark too                                   -> ratio near 1.
    Peaks are searched on each side of the valley wherever they are, so narrow
    knees are found too (fixed strips missed them).
    """
    profile = column_profile(gray)
    v = valley_position(profile)
    left_peak, right_peak = profile[:v].max(initial=0.0), profile[v + 1:].max(initial=0.0)
    return float(profile[v] / (min(left_peak, right_peak) + 1e-6))


def split_knees(gray: np.ndarray, max_gap_ratio: float = 0.6) -> list[np.ndarray]:
    """Return [left knee, right knee] for a bilateral image, else [the image].

    The cut is placed at the darkest column in the middle third, not blindly at the centre.
    The right knee is mirrored so every knee has the same orientation.
    """
    gray = crop_to_content(gray)
    h, w = gray.shape
    if gap_ratio(gray) >= max_gap_ratio:
        return [gray]  # no dark gap between two bright sides: a single knee
    cut = valley_position(column_profile(gray))
    left, right = crop_to_content(gray[:, :cut]), crop_to_content(gray[:, cut:])
    if min(left.shape[1], right.shape[1]) < 0.2 * w:
        return [gray]  # one side nearly empty: not really two knees
    return [left, np.ascontiguousarray(right[:, ::-1])]
