"""Shared pytest fixtures.

Provides a synthetic film-negative fixture: a known ground-truth positive
image (gradients + solid color patches, with a black border simulating the
unexposed rebate) is forward-modeled into a synthetic color-negative scan
and saved as tiny TIFF/JPEG files under tests/data/synthetic/. The inverter
is expected to recover the ground-truth positive from these files.

Forward model -- the exact inverse of ``invert.py``'s density mode with
``paper_black = 0`` and ``scan_exposure_ev = 0``:

    printed = positive ** (1 / grade)
    light   = 1 - 0.9 * printed
    t       = base_rgb * light ** dmax        # == base * 10 ** -D

Bright subjects become dense (low t); a positive of 0 gives t == base, which
is why the rebate border is black: unexposed film is clear film base, and
clear film prints black.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from filmneg.engine.io import Image, export
from filmneg.engine.settings import FilmSettings, OutputSettings

SYNTHETIC_BASE_RGB = (1.0, 0.55, 0.25)  # plausible orange film-base mask
SYNTHETIC_DMAX = 1.8
SYNTHETIC_PAPER_GRADE = 2.0
IMAGE_SIZE = 48
REBATE_MARGIN = 6  # px border simulating unexposed film


@dataclass
class SyntheticNegative:
    """A known ground-truth positive plus its forward-modeled negative files."""

    positive: np.ndarray  # float32 HxWx3 in [0, 1], the ground truth
    negative_rgb: np.ndarray  # float32 HxWx3, the linear negative transmittance
    base_rgb: tuple[float, float, float]
    dmax: float
    paper_grade: float
    rebate_rect: tuple[float, float, float, float]  # normalized (x0, y0, x1, y1)
    tiff_path: Path
    jpeg_path: Path


def _make_ground_truth_positive(size: int, margin: int) -> np.ndarray:
    """Gradients + solid color patches, with a black border (the rebate)."""
    positive = np.zeros((size, size, 3), dtype=np.float32)

    inner = slice(margin, size - margin)
    inner_size = size - 2 * margin

    x = np.linspace(0.0, 1.0, inner_size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, inner_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    positive[inner, inner, 0] = xx
    positive[inner, inner, 1] = yy
    positive[inner, inner, 2] = 1.0 - xx

    # Solid patches (incl. black/white/gray) as known flat reference values,
    # in a strip down the left third so the gradients stay visible beside it.
    strip = slice(margin, margin + inner_size // 3)
    patches = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
    ]
    patch_h = max(inner_size // len(patches), 1)
    for i, color in enumerate(patches):
        row0 = margin + i * patch_h
        row1 = size - margin if i == len(patches) - 1 else row0 + patch_h
        positive[row0:row1, strip, :] = color

    # Rebate border: unexposed film, which prints black (D == 0).
    positive[:margin, :, :] = 0.0
    positive[-margin:, :, :] = 0.0
    positive[:, :margin, :] = 0.0
    positive[:, -margin:, :] = 0.0

    return positive


def _forward_model_negative(
    positive: np.ndarray,
    base_rgb: tuple[float, float, float],
    dmax: float,
    paper_grade: float = SYNTHETIC_PAPER_GRADE,
) -> np.ndarray:
    printed = np.clip(positive, 0.0, 1.0) ** (1.0 / paper_grade)
    light = 1.0 - 0.9 * printed
    base = np.asarray(base_rgb, dtype=np.float32)
    return (base * light**dmax).astype(np.float32)


@pytest.fixture(scope="session")
def make_negative():
    """The forward model, for tests that build their own small negatives."""
    return _forward_model_negative


@pytest.fixture(scope="session")
def synthetic_negative() -> SyntheticNegative:
    positive = _make_ground_truth_positive(IMAGE_SIZE, REBATE_MARGIN)
    negative_rgb = _forward_model_negative(positive, SYNTHETIC_BASE_RGB, SYNTHETIC_DMAX)

    data_dir = Path(__file__).parent / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)

    image = Image(rgb=negative_rgb, source_bit_depth=16, input_encoding="linear")
    tiff_path = data_dir / "negative.tif"
    jpeg_path = data_dir / "negative.jpg"
    export(image, tiff_path, FilmSettings(output=OutputSettings(bit_depth=16, format="tiff")))
    export(image, jpeg_path, FilmSettings(output=OutputSettings(jpeg_quality=95, format="jpg")))

    margin_frac = REBATE_MARGIN / IMAGE_SIZE
    rebate_rect = (0.0, 0.0, margin_frac, 1.0)  # left border strip

    return SyntheticNegative(
        positive=positive,
        negative_rgb=negative_rgb,
        base_rgb=SYNTHETIC_BASE_RGB,
        dmax=SYNTHETIC_DMAX,
        paper_grade=SYNTHETIC_PAPER_GRADE,
        rebate_rect=rebate_rect,
        tiff_path=tiff_path,
        jpeg_path=jpeg_path,
    )
