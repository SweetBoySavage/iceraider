"""Land mask from the GSHHG coastline (full resolution), so the ice / water model is never asked about land.

Data: GSHHG 2.3.7 "A Global Self-consistent, Hierarchical, High-resolution Geography Database"
      (Wessel and Smith, 1996), distributed by the University of Hawaii (SOEST) and NOAA. Licence: LGPL.
      File used: gshhs_f.b (96 MB). Download with `download()`; it is not stored in git.

Lakes count as land here: they are not sea, and the model is not meant for them.
A safety margin is added around the coast, because pixels there mix land and sea and the coastline itself
can be off by a few hundred metres in the Arctic.
"""
from __future__ import annotations

import struct
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
COAST_DIR = ROOT / "data" / "reference" / "coast"
GSHHG_FILE = COAST_DIR / "gshhs_f.b"
GSHHG_URL = "https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-bin-2.3.7.zip"
MARGIN_M = 1000.0            # pixels closer than this to land are also masked
LEVEL_LAND = 1               # GSHHG levels: 1 land, 2 lake, 3 island in lake, 4 pond, 5-6 Antarctica
HEADER = struct.Struct(">11i")   # id, n points, flag, west, east, south, north, area, area_full, container, ancestor


def download(path: Path = GSHHG_FILE) -> Path:
    if not path.exists():
        import requests
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.parent / "gshhg-bin.zip"
        r = requests.get(GSHHG_URL, timeout=900)
        r.raise_for_status()
        archive.write_bytes(r.content)
        with zipfile.ZipFile(archive) as z:
            z.extract(path.name, path.parent)
    return path


def read_gshhg(path, min_north: float = -90.0, level: int = LEVEL_LAND) -> list[np.ndarray]:
    """Polygons of one level as arrays of (lon, lat) in degrees. Only those reaching north of `min_north`."""
    data = Path(path).read_bytes()
    polygons, pos = [], 0
    while pos < len(data):
        _id, n, flag, _west, _east, _south, north, *_ = HEADER.unpack_from(data, pos)
        pos += HEADER.size
        if (flag & 255) == level and north / 1e6 >= min_north:
            polygons.append(np.frombuffer(data, ">i4", 2 * n, pos).reshape(n, 2) / 1e6)
        pos += 8 * n
    if pos != len(data):
        raise ValueError(f"{path}: file ended in the middle of a polygon")
    return polygons


@lru_cache(maxsize=2)
def _arctic_land(path: str, epsg: int):
    """Land polygons north of 55 N, projected. Returns (list of xy arrays, bounds array)."""
    to_map = Transformer.from_crs(4326, epsg, always_xy=True)
    projected = []
    for lonlat in read_gshhg(path, min_north=55.0):
        keep = lonlat[:, 1] > 40.0                        # big continents reach far south; cut them off there
        if keep.sum() < 3:
            continue
        x, y = to_map.transform(lonlat[keep, 0], lonlat[keep, 1])
        projected.append(np.column_stack([x, y]))
    bounds = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in projected])
    return projected, bounds


def rasterize_land(polygons: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where a pixel centre lies inside any polygon. x ascending, y descending, square pixels."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    spacing = float(x[1] - x[0])
    shapes = [({"type": "Polygon", "coordinates": [p.tolist()]}, 1) for p in polygons]
    if not shapes:
        return np.zeros((len(y), len(x)), bool)
    return rasterize(shapes, out_shape=(len(y), len(x)), fill=0, dtype="uint8",
                     transform=from_origin(x[0] - spacing / 2, y[0] + spacing / 2, spacing, spacing)).astype(bool)


def add_margin(land: np.ndarray, spacing_m: float, margin_m: float = MARGIN_M) -> np.ndarray:
    """Land plus every pixel within `margin_m` of it."""
    if margin_m <= 0 or not land.any():
        return land.copy()
    return distance_transform_edt(~land) * spacing_m <= margin_m


def land_mask(x: np.ndarray, y: np.ndarray, epsg: int, margin_m: float = MARGIN_M, path: Path = GSHHG_FILE) -> np.ndarray:
    """Boolean grid (len(y), len(x)): True = land or within the safety margin of land."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing. Run: python -c 'from scripts.land_mask import download; download()'")
    polygons, bounds = _arctic_land(str(path), epsg)
    pad = margin_m + abs(float(x[1] - x[0]))
    x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
    near = (bounds[:, 2] >= x0) & (bounds[:, 0] <= x1) & (bounds[:, 3] >= y0) & (bounds[:, 1] <= y1)
    land = rasterize_land([p for p, hit in zip(polygons, near) if hit], x, y)
    return add_margin(land, abs(float(x[1] - x[0])), margin_m)
