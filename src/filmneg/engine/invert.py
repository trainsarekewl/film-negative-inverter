"""Negative -> positive inversion.

Two models, selected by ``settings.invert.mode``:

* ``"linear"`` -- normalize by the base then ``1 - t``. Simple and fast, but
  ignores that film density is logarithmic, so midtones come out wrong. Kept
  as a correctness stepping stone and a selectable mode.
* ``"density"`` (default) -- darktable negadoctor's print model. Convert
  transmittance to density, then simulate printing the negative onto paper:

      t     = x / base                      # base -> 1
      D     = -log10(t) + ev * log10(2)     # density above the base
      light = 10 ** (-D / dmax)             # light reaching the paper, 1 .. 0.1
      print = (1 + black - light) / (0.9 + black)
      out   = max(print, 0) ** grade

  Clear film (the base, D = 0) lets all the light through and prints black;
  film at ``dmax`` passes a tenth of it and prints white. The 0.9 denominator
  (``1 - 10**-1``) normalizes so D == dmax lands exactly on 1.0 when black is 0.

``film_type`` picks what gets inverted: ``color_neg`` per channel, ``bw_neg``
from one shared density, ``slide`` not at all.
"""

from __future__ import annotations

import numpy as np

from filmneg.engine.settings import FilmSettings, InvertSettings

_EPS = 1e-6
_LOG10_2 = float(np.log10(2.0))
_LIGHT_AT_DMAX = 0.1  # 10 ** (-dmax / dmax)


def invert(rgb: np.ndarray, settings: FilmSettings) -> np.ndarray:
    """Invert a linear float32 HxWx3 scan into a linear positive."""
    film_type = settings.film_type
    if film_type == "slide":
        return rgb.astype(np.float32, copy=True)
    if film_type not in ("color_neg", "bw_neg"):
        raise ValueError(f"Unknown film_type: {film_type!r}")
    if settings.base_rgb is None:
        raise ValueError("base_rgb is not set -- sample the film base before inverting")

    base = np.asarray(settings.base_rgb, dtype=np.float32)
    transmittance = np.clip(rgb / np.maximum(base, _EPS), _EPS, 1.0)

    params = settings.invert
    if params.mode == "linear":
        if film_type == "bw_neg":
            transmittance = _shared_channel(transmittance)
        return (1.0 - transmittance).astype(np.float32)
    if params.mode == "density":
        density = -np.log10(transmittance)
        if film_type == "bw_neg":
            density = _shared_channel(density)
        return _print_density(density, params)
    raise ValueError(f"Unknown invert mode: {params.mode!r}")


def _shared_channel(values: np.ndarray) -> np.ndarray:
    """Collapse to one channel (equal-weight mean) and broadcast back to 3.

    Averaging *after* base normalization cancels any tint in a B&W base or
    the scanner, so the result is exactly neutral.
    """
    mean = values.mean(axis=-1, keepdims=True)
    return np.repeat(mean, 3, axis=-1)


def _print_density(density: np.ndarray, params: InvertSettings) -> np.ndarray:
    density = density + params.scan_exposure_ev * _LOG10_2
    light = 10.0 ** (-density / params.dmax)
    black = params.paper_black
    printed = (1.0 + black - light) / (1.0 - _LIGHT_AT_DMAX + black)
    return (np.maximum(printed, 0.0) ** params.paper_grade).astype(np.float32)
