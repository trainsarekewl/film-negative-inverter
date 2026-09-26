from __future__ import annotations

import numpy as np
import pytest

from filmneg.engine.base import auto_base, sample_base
from filmneg.engine.io import load_linear


def test_sample_base_recovers_base_from_rebate(synthetic_negative):
    base = sample_base(synthetic_negative.negative_rgb, synthetic_negative.rebate_rect)
    np.testing.assert_allclose(base, synthetic_negative.base_rgb, atol=1e-6)


def test_sample_base_after_tiff_round_trip(synthetic_negative):
    image = load_linear(synthetic_negative.tiff_path)
    base = sample_base(image.rgb, synthetic_negative.rebate_rect)
    np.testing.assert_allclose(base, synthetic_negative.base_rgb, atol=1e-3)


def test_sample_base_median_ignores_dust():
    patch = np.full((20, 20, 3), (0.8, 0.5, 0.3), dtype=np.float32)
    patch[3, 4] = 0.0
    patch[10, 10] = 0.05
    patch[15, 2] = 1.0
    assert sample_base(patch, (0.0, 0.0, 1.0, 1.0)) == pytest.approx((0.8, 0.5, 0.3))


def test_sample_base_tiny_rect_still_samples_a_pixel():
    rgb = np.full((10, 10, 3), 0.5, dtype=np.float32)
    assert sample_base(rgb, (0.5, 0.5, 0.5001, 0.5001)) == pytest.approx((0.5, 0.5, 0.5))


@pytest.mark.parametrize(
    "rect", [(0.5, 0.0, 0.5, 1.0), (0.0, 0.0, 1.2, 1.0), (-0.1, 0.0, 0.5, 1.0)]
)
def test_sample_base_rejects_invalid_rect(rect):
    with pytest.raises(ValueError):
        sample_base(np.zeros((4, 4, 3), dtype=np.float32), rect)


def test_auto_base_finds_rebate(synthetic_negative):
    base = auto_base(synthetic_negative.negative_rgb)
    np.testing.assert_allclose(base, synthetic_negative.base_rgb, atol=1e-6)


def test_auto_base_ignores_clipped_backlight(synthetic_negative):
    rgb = synthetic_negative.negative_rgb.copy()
    rgb[:4, :, :] = 1.0
    base = auto_base(rgb)
    np.testing.assert_allclose(base, synthetic_negative.base_rgb, atol=1e-6)
