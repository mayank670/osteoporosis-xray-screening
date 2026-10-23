"""Deterministic preprocessing: decoded array -> model-ready array.

Order of operations (see docs/preprocessing.md for the reasoning):

    1. to_grayscale        X-rays carry one intensity; colour is noise
    2. to_uint8            map 8- or 16-bit intensities onto 0..255
    3. apply_clahe         optional contrast enhancement (off by default)
    4. letterbox_resize    pad to square, then resize (no distortion)
    5. to_model_input      repeat to 3 channels, float32 in [0, 255]

The output is NOT divided by 255. Each model normalises its own input inside
the network (EfficientNet has built-in Rescaling/Normalization layers).
"""

from __future__ import annotations

import cv2
import numpy as np

from config import PREPROCESSING, PreprocessingConfig


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """(H, W) stays as is; (H, W, 3) RGB is converted with luminance weights."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Expected (H, W) or (H, W, 3), got {image.shape}")


def to_uint8(gray: np.ndarray) -> np.ndarray:
    """Map intensities onto 0..255 without assuming the bit depth.

    uint8: already 0..255, returned unchanged.
    uint16: medical images often use only 10-12 of the 16 bits, so dividing
            by 65535 would leave a nearly black image. We stretch the image's
            own min..max onto 0..255 instead (min-max scaling).
    """
    if gray.dtype == np.uint8:
        return gray
    if gray.dtype != np.uint16:
        raise ValueError(f"Expected uint8 or uint16, got {gray.dtype}")
    low, high = int(gray.min()), int(gray.max())
    if high == low:  # constant image: avoid division by zero
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = (gray.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid: tuple[int, int]) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on a uint8 image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def letterbox_resize(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pad the shorter side with black to make a square, then resize to `size` (H, W).

    Padding first keeps the knee's proportions; stretching would distort bone
    shapes differently for every input aspect ratio.
    """
    height, width = gray.shape
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    square = cv2.copyMakeBorder(
        gray, top, side - height - top, left, side - width - left,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    return stretch_resize(square, size)


def stretch_resize(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to `size` = (H, W). Note cv2.resize takes (W, H)."""
    target_h, target_w = size
    shrinking = gray.shape[0] > target_h or gray.shape[1] > target_w
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(gray, (target_w, target_h), interpolation=interpolation)


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (H, W, 3) float32 in [0, 255] by repeating the channel."""
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2).astype(np.float32)


def _geometry(gray: np.ndarray, size: tuple[int, int], cfg: PreprocessingConfig) -> np.ndarray:
    return letterbox_resize(gray, size) if cfg.keep_aspect_ratio else stretch_resize(gray, size)


def _intensity(image: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    gray = to_uint8(to_grayscale(image))
    if cfg.use_clahe:
        gray = apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    return gray


def preprocess_image(image: np.ndarray, cfg: PreprocessingConfig = PREPROCESSING) -> np.ndarray:
    """Decoded image -> (H, W, 3) float32 array in [0, 255], ready for the model."""
    return to_model_input(_geometry(_intensity(image, cfg), cfg.image_size, cfg))


def prepare_display_image(
    image: np.ndarray, display_size: int, cfg: PreprocessingConfig = PREPROCESSING
) -> np.ndarray:
    """Same intensity + geometry steps as the model, at a larger size, as uint8 (H, W).

    Grad-CAM heatmaps are computed in model-input coordinates; drawing them on
    this image lines them up exactly, because both share the same padding.
    """
    return _geometry(_intensity(image, cfg), (display_size, display_size), cfg)
