"""The whole conversion as one function, shared verbatim by batch CLI and GUI.

Stage order: geometry -> invert -> tone -> color -> dust.

Tone runs *before* color: auto levels stretch each channel independently,
which would exactly cancel any white-balance gains applied ahead of it. With
levels first, they remove the film's cast and WB/temperature/tint stay
meaningful creative adjustments on top.

Output encoding (linear -> sRGB, quantizing) is not a stage here: the result
stays linear, and ``io.export`` / the GUI display encode it at the edge.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from filmneg.engine.base import auto_base, sample_base
from filmneg.engine.color import apply_color
from filmneg.engine.geometry import apply_geometry
from filmneg.engine.invert import invert
from filmneg.engine.settings import FilmSettings
from filmneg.engine.tone import apply_tone

PROXY_LONG_EDGE = 2048


def process(rgb: np.ndarray, settings: FilmSettings, *, proxy: bool = False) -> np.ndarray:
    """Convert a linear float32 HxWx3 scan into a linear float32 positive.

    ``rgb`` is never modified. ``proxy=True`` downscales right after geometry
    so a live preview stays fast; every setting is resolution-relative, so the
    proxy and the full-res export produce the same look.
    """
    if settings.dust.enabled:
        raise NotImplementedError("Dust removal arrives in Milestone 5")

    if settings.base_rgb is None and settings.film_type != "slide":
        settings = replace(settings, base_rgb=resolve_base(rgb, settings))

    scan = apply_geometry(rgb, settings.geometry)
    if proxy:
        scan = downscale(scan, PROXY_LONG_EDGE)

    # invert always returns a fresh array, so later stages can work in place.
    positive = invert(scan, settings)
    apply_tone(positive, settings.tone)
    apply_color(positive, settings.color)
    return positive


def resolve_base(rgb: np.ndarray, settings: FilmSettings) -> tuple[float, float, float]:
    """The rebate rect if one is set, else the brightest-pixels fallback.

    Sampled on the scan *before* geometry, so the rect is in original-scan
    coordinates and a crop that trims the rebate away can't lose it.
    """
    rect = settings.geometry.base_sample_rect
    return sample_base(rgb, rect) if rect is not None else auto_base(rgb)


def downscale(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    """Box-filter by an integer factor until the long edge fits ``long_edge``."""
    factor = math.ceil(max(rgb.shape[:2]) / long_edge)
    if factor <= 1:
        return rgb
    height = rgb.shape[0] // factor
    width = rgb.shape[1] // factor
    blocks = rgb[: height * factor, : width * factor].reshape(height, factor, width, factor, 3)
    return blocks.mean(axis=(1, 3), dtype=np.float32)
