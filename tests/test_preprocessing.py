import dataclasses

import numpy as np
import pytest

from config import PREPROCESSING
from ml.preprocessing import InvalidImageError, decode_image, preprocess_image, prepare_display_image
from ml.preprocessing.transforms import letterbox_resize, to_grayscale, to_uint8
from tests.conftest import encode

# ---------- decoding ----------

def test_decode_8bit_gray_png(gray8):
    out = decode_image(encode(gray8))
    assert out.shape == (300, 200) and out.dtype == np.uint8
    assert np.array_equal(out, gray8)  # PNG is lossless


def test_decode_preserves_16bit(gray16):
    out = decode_image(encode(gray16))
    assert out.dtype == np.uint16
    assert out.max() == 4000  # NOT clipped to 255


def test_decode_rgb_returns_rgb_order():
    image = np.zeros((100, 100, 3), np.uint8)
    image[..., 0] = 200  # red channel only
    out = decode_image(encode(image))
    assert out.shape == (100, 100, 3)
    assert out[..., 0].mean() == 200 and out[..., 2].mean() == 0


def test_decode_jpeg(gray8):
    assert decode_image(encode(gray8, ".jpg")).shape == (300, 200)


@pytest.mark.parametrize("data", [b"", b"hello world", b"%PDF-1.4 fake"])
def test_unsupported_content_rejected(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_corrupt_image_rejected(png_bytes):
    with pytest.raises(InvalidImageError, match="corrupt"):
        decode_image(png_bytes[:40])  # valid signature, truncated body


def test_extension_is_irrelevant_only_content_counts(png_bytes):
    # A text file renamed to .png is still text: decode_image never sees a name.
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image" + png_bytes)


def test_tiny_image_rejected():
    with pytest.raises(InvalidImageError, match="too small"):
        decode_image(encode(np.zeros((10, 10), np.uint8)))


def test_png_decompression_bomb_rejected_from_header(png_bytes):
    bomb = bytearray(png_bytes)
    bomb[16:20] = (100_000).to_bytes(4, "big")  # claim 100000 x 100000 pixels
    bomb[20:24] = (100_000).to_bytes(4, "big")
    with pytest.raises(InvalidImageError, match="too large"):
        decode_image(bytes(bomb))

# ---------- intensity ----------

def test_to_uint8_leaves_8bit_unchanged(gray8):
    assert to_uint8(gray8) is gray8


def test_to_uint8_stretches_16bit_to_full_range(gray16):
    out = to_uint8(gray16)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_to_uint8_constant_image_is_safe():
    out = to_uint8(np.full((64, 64), 1234, np.uint16))
    assert out.dtype == np.uint8 and out.max() == 0


def test_to_uint8_rejects_float():
    with pytest.raises(ValueError):
        to_uint8(np.zeros((64, 64), np.float32))


def test_rgb_and_gray_give_identical_model_input(gray8, rgb8):
    assert np.array_equal(preprocess_image(gray8), preprocess_image(rgb8))


def test_to_grayscale_rejects_bad_layout():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((64, 64, 2), np.uint8))

# ---------- geometry ----------

def test_letterbox_keeps_aspect_ratio():
    tall = np.full((400, 200), 255, np.uint8)  # white, twice as tall as wide
    out = letterbox_resize(tall, (100, 100))
    assert out.shape == (100, 100)
    assert out[:, :20].max() == 0 and out[:, -20:].max() == 0  # black side bars
    assert out[:, 30:70].min() == 255  # the image itself, centred


def test_non_square_target_uses_height_width_order(gray8):
    cfg = dataclasses.replace(PREPROCESSING, image_size=(120, 160))
    assert preprocess_image(gray8, cfg).shape == (120, 160, 3)

# ---------- full pipeline ----------

@pytest.mark.parametrize("fixture", ["gray8", "gray16", "rgb8"])
def test_model_input_contract(fixture, request):
    out = preprocess_image(request.getfixturevalue(fixture))
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 255.0
    assert out.max() > 1.0  # proves we did NOT also divide by 255
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_preprocessing_is_deterministic(gray16):
    assert np.array_equal(preprocess_image(gray16), preprocess_image(gray16))


def test_clahe_option_changes_pixels_but_not_contract(gray8):
    cfg = dataclasses.replace(PREPROCESSING, use_clahe=True)
    with_clahe = preprocess_image(gray8, cfg)
    assert with_clahe.shape == (224, 224, 3) and with_clahe.dtype == np.float32
    assert not np.array_equal(with_clahe, preprocess_image(gray8))


def test_display_image_matches_model_geometry(gray8):
    display = prepare_display_image(gray8, 448)
    assert display.shape == (448, 448) and display.dtype == np.uint8
    model_view = preprocess_image(gray8)[..., 0]
    # Downscaling the display image reproduces the model view (same padding).
    small = letterbox_resize(display, (224, 224)).astype(np.float32)
    assert np.abs(small - model_view).mean() < 2.0
