"""White balance, temperature/tint, and the sRGB transfer functions."""

from __future__ import annotations

import numpy as np

from filmneg.engine.geometry import rect_slices
from filmneg.engine.settings import ColorSettings


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function. ``rgb`` is float in [0, 1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer function. ``rgb`` is linear-light float in [0, 1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def color_gains(color: ColorSettings) -> np.ndarray:
    """Fold white balance, temperature and tint into one per-channel multiplier.

    ``temperature`` is in stops along the blue-amber axis (+ warmer: red up and
    blue down by half a stop each per stop). ``tint`` is in stops along the
    green-magenta axis (+ more magenta: green down).
    """
    red, green, blue = color.wb_gains
    warm = 2.0 ** (color.temperature / 2.0)
    magenta = 2.0**color.tint
    return np.array([red * warm, green / magenta, blue / warm], dtype=np.float32)


def apply_color(rgb: np.ndarray, color: ColorSettings) -> np.ndarray:
    """Multiply by :func:`color_gains`, in place."""
    rgb *= color_gains(color)
    return rgb


def neutral_gains(
    rgb: np.ndarray, rect: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    """White-balance gains that make the patch at ``rect`` grey (green stays 1).

    Computed against ``rgb`` as given, so pass the image *before* colour gains
    are applied and store the result in ``ColorSettings.wb_gains``.
    """
    rows, cols = rect_slices(rect, *rgb.shape[:2])
    patch = np.median(rgb[rows, cols, :3].reshape(-1, 3), axis=0)
    if (patch <= 0).any():
        raise ValueError("Neutral patch is black in some channel; pick a brighter grey")
    red, green, blue = (float(patch[1] / value) for value in patch)
    return (red, green, blue)
