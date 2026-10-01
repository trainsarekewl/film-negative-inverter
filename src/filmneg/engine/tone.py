"""Levels and contrast on the linear positive.

The apply_* functions modify ``rgb`` in place (and return it): a full-res
float32 RAW is ~300 MB per copy, so the pipeline avoids making new ones.
"""

from __future__ import annotations

import numpy as np

from filmneg.engine.settings import ToneSettings

MIDDLE_GREY = 0.18
AUTO_BLACK_PERCENTILE = 0.1
AUTO_WHITE_PERCENTILE = 99.9
# Percentiles are estimated on a strided subsample of at most ~1M pixels:
# statistically the same answer, far cheaper, and it keeps proxy and full-res
# estimates close.
_AUTO_SAMPLE_PIXELS = 1_000_000
_EPS = 1e-6

RGB = tuple[float, float, float]


def auto_levels(rgb: np.ndarray) -> tuple[RGB, RGB]:
    """Per-channel black and white points from low/high percentiles.

    Stretching each channel between its *own* extremes is what removes the
    colour cast left by the orange mask and unbalanced raw channels: the
    darkest and brightest things in the frame come out neutral.
    """
    stride = max(1, int(np.sqrt(rgb.shape[0] * rgb.shape[1] / _AUTO_SAMPLE_PIXELS)))
    sample = rgb[::stride, ::stride].reshape(-1, 3)
    black = np.percentile(sample, AUTO_BLACK_PERCENTILE, axis=0)
    white = np.percentile(sample, AUTO_WHITE_PERCENTILE, axis=0)
    return _as_rgb(black), _as_rgb(white)


def apply_tone(rgb: np.ndarray, tone: ToneSettings) -> np.ndarray:
    if tone.auto:
        black, white = auto_levels(rgb)
    else:
        black, white = tone.black_point, tone.white_point
    apply_levels(rgb, black, white)
    return apply_contrast(rgb, tone.contrast_decades)


def apply_levels(rgb: np.ndarray, black: RGB, white: RGB) -> np.ndarray:
    """Map ``black`` -> 0 and ``white`` -> 1 per channel. Below-black clips to
    0 (negative light is meaningless); above-white is kept as headroom."""
    black_arr = np.asarray(black, dtype=np.float32)
    span = np.maximum(np.asarray(white, dtype=np.float32) - black_arr, _EPS)
    rgb -= black_arr
    rgb /= span
    return np.maximum(rgb, 0.0, out=rgb)


def apply_contrast(rgb: np.ndarray, contrast: float) -> np.ndarray:
    """Slope in log-log space, pivoting on middle grey.

    Each decade of input brightness spans ``contrast`` decades of output, so
    1.0 is the identity, >1 is punchier, <1 is flatter, and 18% grey never
    moves. Log space is the natural home for this because film density is
    itself a log quantity.
    """
    if contrast <= 0:
        raise ValueError(f"contrast must be > 0, got {contrast}")
    if contrast == 1.0:
        return rgb
    np.maximum(rgb, 0.0, out=rgb)
    rgb /= MIDDLE_GREY
    np.power(rgb, contrast, out=rgb)
    rgb *= MIDDLE_GREY
    return rgb


def _as_rgb(values: np.ndarray) -> RGB:
    return (float(values[0]), float(values[1]), float(values[2]))
