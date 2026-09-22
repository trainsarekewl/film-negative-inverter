from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from filmneg.engine.io import load_linear

RAW_SAMPLES = sorted((Path(__file__).parent / "data" / "raw").glob("*.RAF"))


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_linear("photo.bmp")


@pytest.mark.skipif(not RAW_SAMPLES, reason="no sample RAW files in tests/data/raw/")
@pytest.mark.parametrize("path", RAW_SAMPLES, ids=lambda p: p.name)
def test_raw_load_shape_and_range(path):
    image = load_linear(path)
    assert image.rgb.ndim == 3
    assert image.rgb.shape[2] == 3
    assert image.rgb.dtype == np.float32
    assert image.rgb.min() >= 0.0
    assert image.rgb.max() <= 1.0 + 1e-6
    assert image.source_bit_depth == 16
    assert image.input_encoding == "linear"


@pytest.mark.skipif(not RAW_SAMPLES, reason="no sample RAW files in tests/data/raw/")
def test_raw_proxy_is_half_resolution():
    full = load_linear(RAW_SAMPLES[0])
    proxy = load_linear(RAW_SAMPLES[0], proxy=True)
    assert proxy.rgb.shape[0] == pytest.approx(full.rgb.shape[0] / 2, abs=1)
    assert proxy.rgb.shape[1] == pytest.approx(full.rgb.shape[1] / 2, abs=1)


def test_tiff_round_trip_matches_forward_model(synthetic_negative):
    image = load_linear(synthetic_negative.tiff_path)
    assert image.rgb.shape == synthetic_negative.negative_rgb.shape
    assert image.rgb.dtype == np.float32
    assert image.source_bit_depth == 16
    assert image.input_encoding == "srgb"
    # 16-bit round trip through the sRGB<->linear transfer function should be
    # near lossless.
    max_err = np.abs(image.rgb - synthetic_negative.negative_rgb).max()
    assert max_err < 1e-3


def test_jpeg_round_trip_matches_forward_model_within_lossy_tolerance(synthetic_negative):
    image = load_linear(synthetic_negative.jpeg_path)
    assert image.rgb.shape == synthetic_negative.negative_rgb.shape
    assert image.rgb.dtype == np.float32
    assert image.source_bit_depth == 8
    assert image.input_encoding == "srgb"
    # 8-bit + JPEG DCT compression is lossy, and this fixture's hard patch
    # edges cause ringing that can spike error at individual pixels -- so
    # check mean error (the meaningful signal) with a loose max as a
    # sanity backstop rather than a tight max.
    err = np.abs(image.rgb - synthetic_negative.negative_rgb)
    assert err.mean() < 0.02
    assert err.max() < 0.35


def test_linear_input_encoding_skips_srgb_decode(synthetic_negative):
    image = load_linear(synthetic_negative.tiff_path, input_encoding="linear")
    assert image.input_encoding == "linear"
    # Without undoing sRGB, values should differ from the auto-decoded ones.
    auto = load_linear(synthetic_negative.tiff_path)
    assert not np.allclose(image.rgb, auto.rgb)
