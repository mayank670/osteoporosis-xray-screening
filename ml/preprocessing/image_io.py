"""Safe decoding of image bytes into NumPy arrays.

Nothing here trusts a filename or extension: the *content* decides whether an
image is valid. Both the training data loader and the web upload handler call
`decode_image`, so they accept exactly the same files.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Content signatures ("magic numbers") of the formats we accept.
_SIGNATURES = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*"),
    "bmp": (b"BM",),
}
SUPPORTED_FORMATS = tuple(_SIGNATURES)
SUPPORTED_DTYPES = (np.uint8, np.uint16)
MIN_SIDE = 64  # smaller images cannot carry useful bone texture
MAX_PIXELS = 50_000_000  # refuse "decompression bombs" before they exhaust memory


class InvalidImageError(ValueError):
    """Raised when bytes are not a supported, decodable image."""


def detect_format(data: bytes) -> str:
    """Return the image format named by the leading bytes, or raise."""
    for fmt, signatures in _SIGNATURES.items():
        if any(data.startswith(sig) for sig in signatures):
            return fmt
    raise InvalidImageError(
        f"Unsupported or unrecognised file type. Supported: {', '.join(SUPPORTED_FORMATS).upper()}."
    )


def _check_pixel_count(data: bytes, fmt: str) -> None:
    """Read width/height from the header only, so huge images are rejected cheaply."""
    if fmt == "png" and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width * height > MAX_PIXELS:
            raise InvalidImageError("Image dimensions are too large.")


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into an array of shape (H, W) or (H, W, 3).

    * Bit depth is preserved: 8-bit files give uint8, 16-bit files give uint16.
    * Colour images are returned in RGB order (OpenCV decodes as BGR).
    * An alpha channel, if any, is dropped.
    """
    if not data:
        raise InvalidImageError("The file is empty.")
    fmt = detect_format(data)
    _check_pixel_count(data, fmt)

    buffer = np.frombuffer(data, dtype=np.uint8)  # a view of the bytes, no copy
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise InvalidImageError("The file is corrupt or could not be decoded.")

    if image.dtype not in SUPPORTED_DTYPES:
        raise InvalidImageError(f"Unsupported pixel type {image.dtype}; expected 8- or 16-bit.")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim != 2:
        raise InvalidImageError(f"Unsupported image layout {image.shape}.")

    height, width = image.shape[:2]
    if min(height, width) < MIN_SIDE:
        raise InvalidImageError(f"Image is too small ({width}x{height}); minimum side is {MIN_SIDE}px.")
    if height * width > MAX_PIXELS:
        raise InvalidImageError("Image dimensions are too large.")
    return image


def read_image_file(path: str | Path) -> np.ndarray:
    """Read and decode an image from disk (used by the training pipeline)."""
    return decode_image(Path(path).read_bytes())
