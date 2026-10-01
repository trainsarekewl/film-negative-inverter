from __future__ import annotations

import numpy as np
import pytest

from filmneg.engine.io import load_linear
from filmneg.engine.pipeline import PROXY_LONG_EDGE, downscale, process, resolve_base
from filmneg.engine.settings import (
    DustSettings,
    FilmSettings,
    GeometrySettings,
    InvertSettings,
    ToneSettings,
)


def _settings(synthetic_negative, **overrides) -> FilmSettings:
    invert = InvertSettings(
        dmax=synthetic_negative.dmax, paper_grade=synthetic_negative.paper_grade
    )
    geometry = GeometrySettings(base_sample_rect=synthetic_negative.rebate_rect)
    return FilmSettings(invert=invert, geometry=geometry, **overrides)


def test_recovers_ground_truth_from_tiff_with_default_auto_tone(synthetic_negative):
    """The end-to-end check: file on disk -> base sampled from the rebate ->
    density invert -> auto levels -> colour, and we get the known positive."""
    scan = load_linear(synthetic_negative.tiff_path).rgb
    positive = process(scan, _settings(synthetic_negative))
    error = np.abs(positive - synthetic_negative.positive)
    assert error.mean() < 2e-3
    assert error.max() < 1e-2


def test_recovers_ground_truth_from_lossy_jpeg(synthetic_negative):
    scan = load_linear(synthetic_negative.jpeg_path).rgb
    positive = process(scan, _settings(synthetic_negative))
    assert np.abs(positive - synthetic_negative.positive).mean() < 0.03


def test_does_not_modify_input(synthetic_negative):
    scan = synthetic_negative.negative_rgb.copy()
    process(scan, _settings(synthetic_negative))
    np.testing.assert_array_equal(scan, synthetic_negative.negative_rgb)


def test_does_not_mutate_settings(synthetic_negative):
    settings = _settings(synthetic_negative)
    process(synthetic_negative.negative_rgb, settings)
    assert settings.base_rgb is None


def test_resolve_base_prefers_rebate_rect_then_auto(synthetic_negative):
    with_rect = _settings(synthetic_negative)
    without_rect = FilmSettings()
    for settings in (with_rect, without_rect):
        base = resolve_base(synthetic_negative.negative_rgb, settings)
        np.testing.assert_allclose(base, synthetic_negative.base_rgb, atol=1e-6)


def test_stored_base_is_used_as_is(synthetic_negative):
    wrong_base = tuple(0.5 * c for c in synthetic_negative.base_rgb)
    settings = _settings(synthetic_negative, base_rgb=wrong_base)
    settings.tone.auto = False
    positive = process(synthetic_negative.negative_rgb, settings)
    assert not np.allclose(positive, synthetic_negative.positive, atol=1e-2)


def test_base_rect_is_sampled_before_crop(synthetic_negative):
    settings = _settings(synthetic_negative)
    settings.geometry.crop = (0.25, 0.25, 0.75, 0.75)  # trims the rebate away
    positive = process(synthetic_negative.negative_rgb, settings)
    inner = synthetic_negative.positive[12:36, 12:36]
    assert positive.shape == inner.shape
    settings.tone.auto = False
    manual = process(synthetic_negative.negative_rgb, settings)
    np.testing.assert_allclose(manual, inner, atol=1e-4)


def test_slide_with_neutral_tone_is_identity():
    rgb = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    settings = FilmSettings(film_type="slide", tone=ToneSettings(auto=False))
    np.testing.assert_allclose(process(rgb, settings), rgb, atol=1e-6)


def test_bw_path_is_neutral(synthetic_negative):
    settings = _settings(synthetic_negative, film_type="bw_neg")
    positive = process(synthetic_negative.negative_rgb, settings)
    np.testing.assert_allclose(positive[..., 0], positive[..., 1], atol=1e-5)
    np.testing.assert_allclose(positive[..., 1], positive[..., 2], atol=1e-5)


def test_dust_is_not_implemented_yet(synthetic_negative):
    settings = _settings(synthetic_negative, dust=DustSettings(enabled=True))
    with pytest.raises(NotImplementedError):
        process(synthetic_negative.negative_rgb, settings)


def test_downscale_box_filters_to_fit():
    rgb = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    small = downscale(rgb, 3)
    assert small.shape == (2, 3, 3)
    np.testing.assert_allclose(small[0, 0], rgb[:2, :2].mean(axis=(0, 1)))


def test_downscale_leaves_small_images_alone():
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    assert downscale(rgb, 100) is rgb


def test_proxy_matches_full_res_look(make_negative):
    """Resolution independence: the preview is the full-res result, smaller."""
    width = PROXY_LONG_EDGE * 2
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    positive = np.stack([ramp, ramp**2, 1 - ramp], axis=-1)[np.newaxis].repeat(4, axis=0)
    base = (0.9, 0.55, 0.3)
    negative = make_negative(positive, base, 2.0)
    settings = FilmSettings(base_rgb=base, invert=InvertSettings(dmax=2.0))

    full = process(negative, settings)
    proxy = process(negative, settings, proxy=True)
    assert max(proxy.shape[:2]) <= PROXY_LONG_EDGE
    assert np.abs(proxy - downscale(full, PROXY_LONG_EDGE)).mean() < 2e-3
