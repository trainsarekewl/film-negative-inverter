from __future__ import annotations

import numpy as np
import pytest

from filmneg.engine.geometry import apply_geometry, rect_slices
from filmneg.engine.settings import GeometrySettings

# 2x3 image whose pixel values are their own labels, so orientation is readable.
LABELLED = np.arange(6, dtype=np.float32).reshape(2, 3, 1).repeat(3, axis=-1)


def _labels(rgb: np.ndarray) -> list[list[float]]:
    return rgb[..., 0].tolist()


def test_identity_is_a_view():
    out = apply_geometry(LABELLED, GeometrySettings())
    assert np.shares_memory(out, LABELLED)
    assert _labels(out) == [[0, 1, 2], [3, 4, 5]]


def test_rotate_90_is_clockwise():
    out = apply_geometry(LABELLED, GeometrySettings(rotate_degrees=90))
    assert _labels(out) == [[3, 0], [4, 1], [5, 2]]


@pytest.mark.parametrize("degrees", [-90, 270])
def test_rotate_counterclockwise(degrees):
    out = apply_geometry(LABELLED, GeometrySettings(rotate_degrees=degrees))
    assert _labels(out) == [[2, 5], [1, 4], [0, 3]]


def test_fine_angle_rotation_is_rejected():
    with pytest.raises(ValueError, match="multiple of 90"):
        apply_geometry(LABELLED, GeometrySettings(rotate_degrees=1.5))


def test_flips():
    assert _labels(apply_geometry(LABELLED, GeometrySettings(flip_horizontal=True))) == [
        [2, 1, 0],
        [5, 4, 3],
    ]
    assert _labels(apply_geometry(LABELLED, GeometrySettings(flip_vertical=True))) == [
        [3, 4, 5],
        [0, 1, 2],
    ]


def test_crop_applies_after_rotation():
    # After a clockwise turn the image is 3 rows x 2 cols; keep the top row.
    geometry = GeometrySettings(rotate_degrees=90, crop=(0.0, 0.0, 1.0, 1 / 3))
    assert _labels(apply_geometry(LABELLED, geometry)) == [[3, 0]]


def test_rect_slices_never_empty():
    rows, cols = rect_slices((0.5, 0.5, 0.5001, 0.5001), 10, 10)
    assert rows.stop - rows.start == 1
    assert cols.stop - cols.start == 1


@pytest.mark.parametrize(
    "rect", [(0.5, 0.0, 0.5, 1.0), (0.0, 0.0, 1.2, 1.0), (-0.1, 0.0, 0.5, 1.0)]
)
def test_rect_slices_rejects_invalid(rect):
    with pytest.raises(ValueError):
        rect_slices(rect, 4, 4)
