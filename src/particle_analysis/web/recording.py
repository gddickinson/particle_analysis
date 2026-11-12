"""Reading an uploaded recording for the local web app.

The movie itself is read by the package's io.readers.DataReader.read_movie (as in the desktop app). The physical pixel
size is looked up in the file's own metadata (OME PhysicalSizeX, then ImageJ/TIFF resolution in microns); when the
file has none, the caller falls back to DiffusionAnalyzer's default and the result says so.
"""
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..io.readers import DataReader
from .pipeline import PipelineError

UM = {"um": 1.0, "µm": 1.0, "\\u00B5m": 1.0, "micron": 1.0, "nm": 1e-3, "mm": 1e3}


def _ome_pixel_size(tif) -> Optional[float]:
    import xml.etree.ElementTree as ET
    xml = getattr(tif, "ome_metadata", None)
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for el in root.iter():
        if el.tag.endswith("Pixels") and el.get("PhysicalSizeX"):
            unit = el.get("PhysicalSizeXUnit", "µm")
            return float(el.get("PhysicalSizeX")) * UM.get(unit, 1.0)
    return None


def _imagej_pixel_size(tif) -> Optional[float]:
    ij = getattr(tif, "imagej_metadata", None) or {}
    unit = str(ij.get("unit", "")).replace("\\u00B5", "µ")
    tag = tif.pages[0].tags.get("XResolution")
    if unit in UM and tag is not None:
        num, den = tag.value
        if num:
            return den / num * UM[unit]
    return None


def pixel_size_from_tiff(path: Path) -> Tuple[Optional[float], str]:
    """(pixel size in um or None, where it came from)."""
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            v = _ome_pixel_size(tif)
            if v:
                return v, "OME metadata (PhysicalSizeX)"
            v = _imagej_pixel_size(tif)
            if v:
                return v, "ImageJ metadata (resolution and unit)"
    except Exception:  # unreadable metadata: fall back, the movie reader reports real read errors
        pass
    return None, "not in the file"


def read_upload(data: bytes, filename: str, max_frames: int) -> Tuple[np.ndarray, Dict]:
    """Movie (t, y, x) from uploaded TIFF bytes, cut to max_frames, plus metadata about it."""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".tif", ".tiff"):
        raise PipelineError("Upload a TIFF stack (.tif or .tiff), as the desktop app reads.")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ("upload" + suffix)
        path.write_bytes(data)
        movie = DataReader().read_movie(path)
        px, px_source = pixel_size_from_tiff(path)
    if movie is None:
        raise PipelineError("Could not read the file as a TIFF stack.")
    if movie.ndim == 2:
        movie = movie[None]
    if movie.ndim != 3:
        raise PipelineError(f"Expected a (frames, height, width) stack, got shape {movie.shape}.")
    n_total = movie.shape[0]
    movie = movie[:max_frames]
    return movie, {"n_frames_total": int(n_total), "n_frames_used": int(movie.shape[0]),
                   "shape": list(movie.shape), "dtype": str(movie.dtype),
                   "pixel_size_um": px, "pixel_size_source": px_source}
