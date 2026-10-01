"""Format-agnostic image I/O: load into one linear float32 RGB representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from filmneg.engine.color import linear_to_srgb, srgb_to_linear
from filmneg.engine.settings import FilmSettings

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}

# dtype -> full-scale value used to normalize integer sample data to [0, 1].
_MAX_VALUE_BY_ITEMSIZE = {1: 255.0, 2: 65535.0, 4: 4294967295.0}


@dataclass
class Image:
    """One linear float32 RGB image plus enough metadata to export it back out."""

    rgb: np.ndarray  # float32, HxWx3, linear light, nominally [0, 1]
    source_bit_depth: int
    input_encoding: str  # "linear" | "srgb" -- encoding of the source pixels
    exif: dict[str, Any] = field(default_factory=dict)


def load_linear(path: str | Path, *, proxy: bool = False, input_encoding: str = "auto") -> Image:
    """Dispatch by extension into one linear float32 RGB :class:`Image`.

    ``proxy=True`` requests a fast, lower-resolution decode for preview use
    where the format supports it (currently: RAW half-size demosaic).

    ``input_encoding`` controls whether TIFF/JPEG source samples are treated
    as sRGB-encoded (the default assumption, and the only sane one for JPEG)
    or already linear (e.g. a TIFF produced by another linear-light tool).
    RAW is always linear straight out of libraw, so this is ignored for RAW.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return _load_raw(path, proxy=proxy)
    if ext in TIFF_EXTENSIONS:
        return _load_tiff(path, input_encoding=input_encoding)
    if ext in JPEG_EXTENSIONS:
        return _load_jpeg(path, input_encoding=input_encoding)
    raise ValueError(f"Unsupported file extension: {ext!r}")


def export(image: Image, path: str | Path, settings: FilmSettings) -> None:
    """Encode ``image.rgb`` (linear) per ``settings.output`` and write it out.

    EXIF is copied best-effort: only the piexif-shaped dict produced by
    :func:`_load_jpeg` can be re-embedded, and only into a JPEG output --
    everything else (RAW's flat exifread dict, TIFF's tag dict, or no exif
    at all) is silently dropped rather than failing the export.
    """
    path = Path(path)
    ext = path.suffix.lower()

    encoded = (
        image.rgb if settings.output.color_space == "linear" else linear_to_srgb(image.rgb)
    )

    if ext in TIFF_EXTENSIONS:
        _export_tiff(encoded, path, settings)
    elif ext in JPEG_EXTENSIONS:
        _export_jpeg(encoded, path, settings, image.exif)
    else:
        raise ValueError(f"Unsupported export extension: {ext!r}")


def _quantize(rgb: np.ndarray, max_value: int) -> np.ndarray:
    dtype = np.uint8 if max_value <= 255 else np.uint16
    return np.clip(rgb * max_value + 0.5, 0, max_value).astype(dtype)


def _export_tiff(rgb: np.ndarray, path: Path, settings: FilmSettings) -> None:
    import tifffile

    bit_depth = settings.output.bit_depth
    if bit_depth not in (8, 16):
        raise ValueError(f"Unsupported TIFF export bit depth: {bit_depth}")
    tifffile.imwrite(str(path), _quantize(rgb, 2**bit_depth - 1))


def _export_jpeg(rgb: np.ndarray, path: Path, settings: FilmSettings, exif: dict[str, Any]) -> None:
    from PIL import Image as PILImage

    data = _quantize(rgb, 255)
    pil_image = PILImage.fromarray(data, mode="RGB")

    save_kwargs: dict[str, Any] = {"quality": settings.output.jpeg_quality}
    exif_bytes = _build_jpeg_exif_bytes(exif)
    if exif_bytes is not None:
        save_kwargs["exif"] = exif_bytes
    pil_image.save(path, "JPEG", **save_kwargs)


_PIEXIF_IFD_KEYS = ("0th", "Exif", "GPS", "1st", "thumbnail")


def _build_jpeg_exif_bytes(exif: dict[str, Any]) -> bytes | None:
    if not exif or not any(key in exif for key in _PIEXIF_IFD_KEYS):
        return None

    try:
        import piexif
    except ImportError:
        return None

    try:
        return piexif.dump(exif)
    except Exception:  # noqa: BLE001 -- best-effort metadata must never fail an export
        return None


def _to_rgb3(array: np.ndarray) -> np.ndarray:
    """Drop alpha/extra samples and broadcast grayscale to 3 channels."""
    if array.ndim == 2:
        return np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1)
    return array[..., :3]


def _resolve_encoding(rgb: np.ndarray, input_encoding: str) -> tuple[np.ndarray, str]:
    """Apply the sRGB->linear transfer function unless told the data already
    is linear. "auto" assumes sRGB, the near-universal default for scanned
    film TIFFs/JPEGs.
    """
    if input_encoding == "linear":
        return rgb, "linear"
    if input_encoding in ("auto", "srgb"):
        return srgb_to_linear(rgb), "srgb"
    raise ValueError(f"Unknown input_encoding: {input_encoding!r}")


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


def _load_tiff(path: Path, *, input_encoding: str) -> Image:
    import tifffile

    array = tifffile.imread(str(path))
    array = _to_rgb3(array)

    if np.issubdtype(array.dtype, np.floating):
        source_bit_depth = array.dtype.itemsize * 8
        rgb = array.astype(np.float32)
    else:
        source_bit_depth = array.dtype.itemsize * 8
        max_value = _MAX_VALUE_BY_ITEMSIZE.get(array.dtype.itemsize)
        if max_value is None:
            raise ValueError(f"Unsupported TIFF sample dtype: {array.dtype}")
        rgb = array.astype(np.float32) / max_value

    rgb, resolved_encoding = _resolve_encoding(rgb, input_encoding)
    return Image(
        rgb=rgb,
        source_bit_depth=source_bit_depth,
        input_encoding=resolved_encoding,
        exif=_extract_tiff_exif(path),
    )


def _extract_tiff_exif(path: Path) -> dict[str, Any]:
    """Best-effort EXIF from the TIFF's own tag directory."""
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            return {tag.name: tag.value for tag in page.tags}
    except Exception:  # noqa: BLE001 -- best-effort metadata must never fail a load
        return {}


def _load_jpeg(path: Path, *, input_encoding: str) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as pil_image:
        array = np.array(pil_image.convert("RGB"))
        exif = _extract_jpeg_exif(pil_image)

    rgb = array.astype(np.float32) / 255.0
    rgb, resolved_encoding = _resolve_encoding(rgb, input_encoding)
    return Image(
        rgb=rgb,
        source_bit_depth=8,
        input_encoding=resolved_encoding,
        exif=exif,
    )


def _extract_jpeg_exif(pil_image: Any) -> dict[str, Any]:
    """Best-effort EXIF via ``piexif``, keyed by IFD name for round-tripping
    back out on export.
    """
    try:
        import piexif
    except ImportError:
        return {}

    raw_exif = pil_image.info.get("exif")
    if not raw_exif:
        return {}
    try:
        return piexif.load(raw_exif)
    except Exception:  # noqa: BLE001 -- best-effort metadata must never fail a load
        return {}
