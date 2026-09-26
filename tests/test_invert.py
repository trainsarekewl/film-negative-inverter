from __future__ import annotations

import numpy as np
import pytest

from filmneg.engine.base import sample_base
from filmneg.engine.invert import invert
from filmneg.engine.io import load_linear
from filmneg.engine.settings import FilmSettings, InvertSettings


def _settings(synthetic_negative, **invert_overrides) -> FilmSettings:
    params = InvertSettings(
        dmax=synthetic_negative.dmax, paper_grade=synthetic_negative.paper_grade
    )
    for key, value in invert_overrides.items():
        setattr(params, key, value)
    return FilmSettings(base_rgb=synthetic_negative.base_rgb, invert=params)


def test_density_mode_recovers_ground_truth(synthetic_negative):
    positive = invert(synthetic_negative.negative_rgb, _settings(synthetic_negative))
    np.testing.assert_allclose(positive, synthetic_negative.positive, atol=1e-4)


def test_density_mode_recovers_ground_truth_from_tiff_with_sampled_base(synthetic_negative):
    image = load_linear(synthetic_negative.tiff_path)
    settings = _settings(synthetic_negative)
    settings.base_rgb = sample_base(image.rgb, synthetic_negative.rebate_rect)
    positive = invert(image.rgb, settings)
    assert np.abs(positive - synthetic_negative.positive).max() < 5e-3


def test_density_mode_endpoints(synthetic_negative):
    base = np.asarray(synthetic_negative.base_rgb, dtype=np.float32)
    at_dmax = base * 10.0 ** (-synthetic_negative.dmax)
    rgb = np.stack([base, at_dmax]).reshape(1, 2, 3)
    positive = invert(rgb, _settings(synthetic_negative))
    np.testing.assert_allclose(positive[0, 0], 0.0, atol=1e-6)  # clear base prints black
    np.testing.assert_allclose(positive[0, 1], 1.0, atol=1e-5)  # dmax prints white


def test_brighter_than_base_clamps_to_black(synthetic_negative):
    rgb = np.full((1, 1, 3), 1.0, dtype=np.float32)
    positive = invert(rgb, _settings(synthetic_negative))
    np.testing.assert_allclose(positive, 0.0, atol=1e-6)


def test_paper_black_lifts_blacks(synthetic_negative):
    rebate = synthetic_negative.negative_rgb[:1, :1]
    lifted = invert(rebate, _settings(synthetic_negative, paper_black=0.1))
    assert (lifted > 0.0).all()


def test_scan_exposure_brightens(synthetic_negative):
    neg = synthetic_negative.negative_rgb
    normal = invert(neg, _settings(synthetic_negative))
    brighter = invert(neg, _settings(synthetic_negative, scan_exposure_ev=1.0))
    assert brighter.mean() > normal.mean()


def test_paper_grade_raises_midtone_contrast(synthetic_negative):
    neg = synthetic_negative.negative_rgb
    soft = invert(neg, _settings(synthetic_negative, paper_grade=1.0))
    hard = invert(neg, _settings(synthetic_negative, paper_grade=3.0))
    midtones = (soft > 0.2) & (soft < 0.8)
    assert (hard[midtones] < soft[midtones]).all()


def test_linear_mode_is_base_normalized_complement():
    base = (0.8, 0.5, 0.3)
    positive = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    negative = np.asarray(base, dtype=np.float32) * (1.0 - positive)
    settings = FilmSettings(base_rgb=base, invert=InvertSettings(mode="linear"))
    np.testing.assert_allclose(invert(negative, settings), positive, atol=1e-5)


@pytest.mark.parametrize("mode", ["density", "linear"])
def test_bw_neg_output_is_neutral_despite_tinted_base(make_negative, mode):
    gray = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4, 1).repeat(3, axis=-1)
    base = (0.82, 0.78, 0.75)  # slightly warm B&W base
    negative = make_negative(gray, base, 2.0, 2.0)
    settings = FilmSettings(
        film_type="bw_neg",
        base_rgb=base,
        invert=InvertSettings(mode=mode, dmax=2.0, paper_grade=2.0),
    )
    positive = invert(negative, settings)
    np.testing.assert_allclose(positive[..., 0], positive[..., 1], atol=1e-6)
    np.testing.assert_allclose(positive[..., 1], positive[..., 2], atol=1e-6)
    if mode == "density":
        np.testing.assert_allclose(positive, gray, atol=1e-4)


def test_slide_is_passthrough_copy():
    rgb = np.random.default_rng(1).random((4, 4, 3), dtype=np.float32)
    out = invert(rgb, FilmSettings(film_type="slide"))
    np.testing.assert_array_equal(out, rgb)
    assert out is not rgb


def test_missing_base_raises():
    with pytest.raises(ValueError, match="base_rgb"):
        invert(np.zeros((2, 2, 3), dtype=np.float32), FilmSettings())


@pytest.mark.parametrize(
    "settings",
    [
        FilmSettings(film_type="polaroid", base_rgb=(1, 1, 1)),
        FilmSettings(base_rgb=(1, 1, 1), invert=InvertSettings(mode="magic")),
    ],
)
def test_unknown_options_raise(settings):
    with pytest.raises(ValueError):
        invert(np.full((2, 2, 3), 0.5, dtype=np.float32), settings)
