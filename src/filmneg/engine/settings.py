"""data object describing a conversion for film negatives

Fully round-trips to a JSON sidecar (``photo.tif`` -> ``photo.filmneg.json``)
so the CLI and GUI share it verbatim. Loading an older or hand-edited sidecar
fills any missing keys with current defaults, so the format is forward- and
backward-compatible as fields are added.
"""

from __future__ import annotations

import json
import types
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

CURRENT_VERSION = 1


@dataclass
class GeometrySettings:
    """Rotate/flip/crop, plus the rebate rect the film base is sampled from.

    ``crop`` and ``base_sample_rect`` are normalized ``(x0, y0, x1, y1)`` in
    [0, 1] so they survive proxy/full-res resolution changes. The base is
    sampled once and cached in ``FilmSettings.base_rgb`` so cropping the
    rebate away later doesn't lose it. ``base_sample_rect`` is in original-scan
    coordinates; ``crop`` is in the rotated/flipped orientation.
    """

    rotate_degrees: float = 0.0  # clockwise, multiples of 90
    flip_horizontal: bool = False
    flip_vertical: bool = False
    crop: tuple[float, float, float, float] | None = None
    base_sample_rect: tuple[float, float, float, float] | None = None


@dataclass
class InvertSettings:
    """Negadoctor-style inversion parameters.

    ``mode`` selects between the density/log model (default) and the plain
    per-channel linear invert kept as a correctness stepping stone.
    """

    mode: str = "density"  # "density" | "linear"
    dmax: float = 2.2
    scan_exposure_ev: float = 0.0
    paper_black: float = 0.0
    paper_grade: float = 2.0


@dataclass
class ToneSettings:
    """Per-channel black/white points (linear) and contrast.

    ``auto`` derives the points from percentiles instead of using the stored
    ones. ``contrast_decades`` is decades of output per decade of input,
    pivoting on middle grey; 1.0 is the identity.
    """

    auto: bool = True
    black_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    white_point: tuple[float, float, float] = (1.0, 1.0, 1.0)
    contrast_decades: float = 1.0


@dataclass
class ColorSettings:
    """Per-channel gains plus temperature (+ warmer) and tint (+ more magenta),
    both in stops."""

    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temperature: float = 0.0
    tint: float = 0.0


@dataclass
class DustSettings:
    """wip"""

    enabled: bool = False


@dataclass
class OutputSettings:
    format: str = "tiff"  # "tiff" | "jpg"
    bit_depth: int = 16
    jpeg_quality: int = 95
    color_space: str = "srgb"


@dataclass
class FilmSettings:
    version: int = CURRENT_VERSION
    film_type: str = "color_neg"  # "color_neg" | "bw_neg" | "slide"
    input_encoding: str = "auto"  # "auto" | "srgb" | "linear"
    base_rgb: tuple[float, float, float] | None = None
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    invert: InvertSettings = field(default_factory=InvertSettings)
    tone: ToneSettings = field(default_factory=ToneSettings)
    color: ColorSettings = field(default_factory=ColorSettings)
    dust: DustSettings = field(default_factory=DustSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> FilmSettings:
        """Parse a JSON string, or read+parse it if ``source`` is a path."""
        if isinstance(source, Path) or (isinstance(source, str) and _looks_like_path(source)):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source
        return _from_dict(cls, json.loads(text))


def _looks_like_path(source: str) -> bool:
    return "{" not in source and Path(source).is_file()


def sidecar_path(image_path: str | Path) -> Path:
    """``photo.tif`` -> ``photo.filmneg.json``."""
    return Path(image_path).with_suffix(".filmneg.json")


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys and leaving
    missing keys at their field defaults (forward/backward compatibility).
    """
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(data[f.name], hints[f.name])
    return cls(**kwargs)


def _coerce(value: Any, type_: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(type_)
    if origin is typing.Union or origin is types.UnionType:
        for arg in get_args(type_):
            if arg is not type(None):
                return _coerce(value, arg)
        return value

    if is_dataclass(type_) and isinstance(value, dict):
        return _from_dict(type_, value)

    if origin is tuple and isinstance(value, list):
        return tuple(value)

    return value
