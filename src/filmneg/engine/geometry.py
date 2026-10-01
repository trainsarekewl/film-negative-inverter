"""Rotate / flip / crop.

Everything here returns numpy views, so geometry costs no pixel copies even
on a full-res RAW.
"""

from __future__ import annotations

import numpy as np

from filmneg.engine.settings import GeometrySettings


def rect_slices(
    rect: tuple[float, float, float, float], height: int, width: int
) -> tuple[slice, slice]:
    """Normalized ``(x0, y0, x1, y1)`` -> ``(row_slice, col_slice)``, never empty."""
    x0, y0, x1, y1 = rect
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Invalid normalized rect: {rect!r}")
    return _span(y0, y1, height), _span(x0, x1, width)


def _span(start: float, end: float, size: int) -> slice:
    first = min(int(np.floor(start * size)), size - 1)
    last = max(int(np.ceil(end * size)), first + 1)
    return slice(first, last)


def apply_geometry(rgb: np.ndarray, geometry: GeometrySettings) -> np.ndarray:
    """Rotate (clockwise), then flip, then crop.

    The crop rect is in the rotated/flipped orientation -- the one the user
    is looking at when they draw it.
    """
    quarter_turns, remainder = divmod(geometry.rotate_degrees, 90)
    if remainder:
        raise ValueError(
            f"rotate_degrees must be a multiple of 90, got {geometry.rotate_degrees}"
        )
    rgb = np.rot90(rgb, k=-int(quarter_turns) % 4)

    if geometry.flip_horizontal:
        rgb = rgb[:, ::-1]
    if geometry.flip_vertical:
        rgb = rgb[::-1]

    if geometry.crop is not None:
        rows, cols = rect_slices(geometry.crop, *rgb.shape[:2])
        rgb = rgb[rows, cols]
    return rgb
