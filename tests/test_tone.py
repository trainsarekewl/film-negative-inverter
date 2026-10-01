from __future__ import annotations

import numpy as np
import pytest

from filmneg.engine.settings import ToneSettings
from filmneg.engine.tone import (
    MIDDLE_GREY,
    apply_contrast,
    apply_levels,
    apply_tone,
    auto_levels,
)


def _ramp() -> np.ndarray:
    return np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape(10, 100, 1).repeat(3, axis=-1)


def test_levels_map_black_and_white_points():
    rgb = np.array([[[0.1, 0.2, 0.3], [0.5, 0.6, 0.9]]], dtype=np.float32)
    apply_levels(rgb, black=(0.1, 0.2, 0.3), white=(0.5, 0.6, 0.9))
    np.testing.assert_allclose(rgb, [[[0, 0, 0], [1, 1, 1]]], atol=1e-6)


def test_levels_clip_below_black_but_keep_highlight_headroom():
    rgb = np.array([[[0.0, 2.0, 0.5]]], dtype=np.float32)
    apply_levels(rgb, black=(0.1, 0.1, 0.1), white=(1.0, 1.0, 1.0))
    assert rgb[0, 0, 0] == 0.0
    assert rgb[0, 0, 1] > 1.0


def test_auto_levels_removes_a_colour_cast():
    cast = _ramp() * np.array([0.9, 0.6, 0.2], dtype=np.float32) + np.array(
        [0.05, 0.1, 0.02], dtype=np.float32
    )
    apply_tone(cast, ToneSettings(auto=True))
    np.testing.assert_allclose(cast[..., 0], cast[..., 1], atol=2e-3)
    np.testing.assert_allclose(cast[..., 1], cast[..., 2], atol=2e-3)


def test_auto_levels_on_full_range_image_is_near_identity():
    black, white = auto_levels(_ramp())
    np.testing.assert_allclose(black, 0.0, atol=2e-3)
    np.testing.assert_allclose(white, 1.0, atol=2e-3)


def test_manual_levels_used_when_auto_is_off():
    rgb = np.full((2, 2, 3), 0.5, dtype=np.float32)
    apply_tone(rgb, ToneSettings(auto=False, black_point=(0.5, 0.5, 0.5)))
    np.testing.assert_allclose(rgb, 0.0)


def test_contrast_one_is_identity():
    rgb = _ramp()
    np.testing.assert_array_equal(apply_contrast(rgb.copy(), 1.0), rgb)


def test_contrast_pivots_on_middle_grey():
    rgb = np.array([[[MIDDLE_GREY, 0.05, 0.8]]], dtype=np.float32)
    apply_contrast(rgb, 1.5)
    assert rgb[0, 0, 0] == pytest.approx(MIDDLE_GREY, rel=1e-5)
    assert rgb[0, 0, 1] < 0.05  # shadows darker
    assert rgb[0, 0, 2] > 0.8  # highlights brighter


def test_contrast_is_a_log_log_slope():
    # One decade below grey should land `contrast` decades below grey.
    rgb = np.full((1, 1, 3), MIDDLE_GREY / 10, dtype=np.float32)
    apply_contrast(rgb, 2.0)
    np.testing.assert_allclose(rgb, MIDDLE_GREY / 100, rtol=1e-5)


def test_non_positive_contrast_raises():
    with pytest.raises(ValueError):
        apply_contrast(_ramp(), 0.0)
