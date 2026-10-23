"""Private file storage with server-generated names.

User-supplied filenames are NEVER used to build a path. Every stored file
gets a random name, and every read is checked to stay inside its base folder.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np


class StoragePathError(ValueError):
    """A stored path points outside its allowed folder (tampering or corruption)."""


def new_name(extension: str) -> str:
    """e.g. '3f2a...9c.png' — 128 random bits, practically unguessable and unique."""
    return f"{uuid.uuid4().hex}.{extension.lstrip('.')}"


def resolve_inside(base: Path, relative: str) -> Path:
    """Join base + relative and refuse anything that escapes base (e.g. '../../secret')."""
    base = Path(base).resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise StoragePathError("Path escapes storage folder")
    return candidate


def save_bytes(base: Path, subfolder: str, data: bytes, extension: str) -> str:
    """Write bytes under base/subfolder/ with a fresh name. Returns the relative path."""
    relative = f"{subfolder}/{new_name(extension)}"
    target = resolve_inside(base, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return relative


def save_png(base: Path, subfolder: str, image: np.ndarray) -> str:
    """Encode a uint8 gray (H, W) or RGB (H, W, 3) array as PNG and save it."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return save_bytes(base, subfolder, buffer.tobytes(), "png")


def delete_quietly(base: Path, relative: str | None) -> bool:
    """Remove a stored file if it exists. Returns True if something was deleted."""
    if not relative:
        return False
    try:
        path = resolve_inside(base, relative)
    except StoragePathError:
        return False
    if path.is_file():
        path.unlink()
        return True
    return False
