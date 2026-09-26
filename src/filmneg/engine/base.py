"""Film-base (Dmin) sampling.

The film base is the clearest part of a negative: the orange-masked rebate
between/around frames that received no exposure. Its transmittance is the
reference every other pixel's density is measured against.
"""

from __future__ import annotations

import numpy as np

# Pixels near-saturated in *every* channel are assumed to be bare backlight
# leaking past the film holder -- neutral white, brighter than the base. The
# orange base itself may clip in red alone, so one clipped channel isn't enough.
_CLIPPED_THRESHOLD = 0.98


def sample_base(
    rgb: np.ndarray, rect: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    """Per-channel median of a rebate patch.

    ``rect`` is normalized ``(x0, y0, x1, y1)`` in [0, 1], so the same rect
    works on a proxy and on the full-res image. The median ignores dust
    specks and scratches inside the patch that a mean would absorb.
    """
    x0, y0, x1, y1 = rect
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Invalid normalized rect: {rect!r}")

    height, width = rgb.shape[:2]
    col0, col1 = _to_pixel_span(x0, x1, width)
    row0, row1 = _to_pixel_span(y0, y1, height)

    patch = rgb[row0:row1, col0:col1, :3].reshape(-1, 3)
    return _as_rgb_tuple(np.median(patch, axis=0))


def auto_base(rgb: np.ndarray, brightest_percent: float = 0.5) -> tuple[float, float, float]:
    """Fallback base estimate when no rebate rect is given: per-channel
    median of the brightest ``brightest_percent`` % of unclipped pixels.

    On a negative the base is the least dense area, i.e. the brightest thing
    in the scan other than raw backlight.
    """
    if not 0.0 < brightest_percent <= 100.0:
        raise ValueError(f"brightest_percent must be in (0, 100], got {brightest_percent}")

    pixels = rgb[..., :3].reshape(-1, 3)
    unclipped = pixels[~(pixels >= _CLIPPED_THRESHOLD).all(axis=1)]
    if unclipped.size:
        pixels = unclipped

    brightness = pixels.mean(axis=1)
    threshold = np.percentile(brightness, 100.0 - brightest_percent)
    return _as_rgb_tuple(np.median(pixels[brightness >= threshold], axis=0))


def _to_pixel_span(start: float, end: float, size: int) -> tuple[int, int]:
    """Normalized span -> pixel indices, never empty."""
    first = min(int(np.floor(start * size)), size - 1)
    last = max(int(np.ceil(end * size)), first + 1)
    return first, last


def _as_rgb_tuple(values: np.ndarray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))
