"""Format-agnostic image I/O: load into one linear float32 RGB representation.

TIFF and JPEG loading land in separate changes; this one implements RAW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass
class Image:
    """One linear float32 RGB image plus enough metadata to export it back out."""

    rgb: np.ndarray  # float32, HxWx3, linear light, nominally [0, 1]
    source_bit_depth: int
    input_encoding: str  # "linear" | "srgb" -- encoding of the source pixels
    exif: dict[str, Any] = field(default_factory=dict)


def load_linear(path: str | Path, *, proxy: bool = False) -> Image:
    """Dispatch by extension into one linear float32 RGB :class:`Image`.

    ``proxy=True`` requests a fast, lower-resolution decode for preview use
    where the format supports it (currently: RAW half-size demosaic).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return _load_raw(path, proxy=proxy)
    if ext in TIFF_EXTENSIONS:
        raise NotImplementedError("TIFF loading is not implemented yet")
    if ext in JPEG_EXTENSIONS:
        raise NotImplementedError("JPEG loading is not implemented yet")
    raise ValueError(f"Unsupported file extension: {ext!r}")


def _load_raw(path: Path, *, proxy: bool) -> Image:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        # gamma=(1, 1) + no_auto_bright + no camera WB: libraw hands back
        # demosaiced sensor data with no tone curve applied, i.e. already
        # linear light, so no further linearization step is needed here.
        rgb16 = raw.postprocess(
            half_size=proxy,
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
        )

    rgb = rgb16.astype(np.float32) / 65535.0
    return Image(
        rgb=rgb,
        source_bit_depth=16,
        input_encoding="linear",
        exif=_extract_raw_exif(path),
    )


def _extract_raw_exif(path: Path) -> dict[str, Any]:
    """Best-effort EXIF via the optional ``exifread`` dependency."""
    try:
        import exifread
    except ImportError:
        return {}

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    return {str(key): str(value) for key, value in tags.items()}
