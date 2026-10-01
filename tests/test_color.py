from __future__ import annotations

import numpy as np
import pytest

from filmneg.engine.color import (
    apply_color,
    color_gains,
    linear_to_srgb,
    neutral_gains,
    srgb_to_linear,
)
from filmneg.engine.settings import ColorSettings


def test_srgb_transfer_functions_round_trip():
    x = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-6)


def test_srgb_encode_known_points():
    assert linear_to_srgb(np.float32(0.18)) == pytest.approx(0.4614, abs=1e-3)
    assert linear_to_srgb(np.float32(1.0)) == pytest.approx(1.0)


def test_default_color_is_identity():
    np.testing.assert_allclose(color_gains(ColorSettings()), (1.0, 1.0, 1.0))


def test_wb_gains_multiply_channels():
    rgb = np.full((2, 2, 3), 0.5, dtype=np.float32)
    apply_color(rgb, ColorSettings(wb_gains=(2.0, 1.0, 0.5)))
    np.testing.assert_allclose(rgb[0, 0], (1.0, 0.5, 0.25))


def test_temperature_warms_and_cools():
    red, green, blue = color_gains(ColorSettings(temperature=1.0))
    assert red > 1.0 > blue
    assert green == 1.0
    assert red / blue == pytest.approx(2.0)  # one stop apart in total

    red, _, blue = color_gains(ColorSettings(temperature=-1.0))
    assert blue > 1.0 > red


def test_tint_moves_green():
    _, green, _ = color_gains(ColorSettings(tint=1.0))
    assert green == pytest.approx(0.5)


def test_neutral_gains_make_patch_grey():
    rgb = np.full((10, 10, 3), (0.4, 0.3, 0.6), dtype=np.float32)
    gains = neutral_gains(rgb, (0.2, 0.2, 0.8, 0.8))
    assert gains[1] == 1.0
    apply_color(rgb, ColorSettings(wb_gains=gains))
    np.testing.assert_allclose(rgb[5, 5], 0.3, atol=1e-6)


def test_neutral_gains_rejects_black_patch():
    with pytest.raises(ValueError):
        neutral_gains(np.zeros((4, 4, 3), dtype=np.float32), (0.0, 0.0, 1.0, 1.0))
