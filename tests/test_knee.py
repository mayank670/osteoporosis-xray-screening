import numpy as np

from ml.preprocessing.knee import content_box, crop_to_content, gap_ratio, split_knees


def canvas(h=300, w=400):
    return np.zeros((h, w), np.uint8)


def test_crop_removes_black_border():
    g = canvas()
    g[50:250, 100:300] = 200
    top, bottom, left, right = content_box(g, margin=0.0)
    assert (top, bottom, left, right) == (50, 250, 100, 300)
    assert crop_to_content(g).shape[0] < g.shape[0]


def test_blank_image_is_kept_whole():
    assert content_box(canvas()) == (0, 300, 0, 400)


def test_two_bright_columns_are_split_and_right_is_mirrored():
    g = canvas()
    g[20:280, 40:150] = 200    # left knee
    g[20:280, 250:360] = 180   # right knee
    g[20:280, 250:270] = 250   # a marker on the right knee's left edge
    assert gap_ratio(crop_to_content(g)) < 0.2
    parts = split_knees(g)
    assert len(parts) == 2
    right = parts[1]
    assert right[:, -25:].max() == 250  # mirrored: the marker moved to the right edge


def test_single_centred_knee_is_not_split():
    g = canvas()
    g[20:280, 150:250] = 200
    assert len(split_knees(g)) == 1


def test_single_off_centre_knee_is_not_split():
    g = canvas()
    g[20:280, 30:170] = 200  # knee on the left, background on the right
    assert len(split_knees(g)) == 1


def test_split_is_deterministic():
    g = canvas()
    g[20:280, 40:150] = 200
    g[20:280, 250:360] = 180
    a, b = split_knees(g), split_knees(g)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
