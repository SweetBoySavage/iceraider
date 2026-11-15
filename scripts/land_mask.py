"""Land mask from the GSHHG coastline (full resolution), so the ice / water model is never asked about land.

Data: GSHHG 2.3.7 "A Global Self-consistent, Hierarchical, High-resolution Geography Database"
      (Wessel and Smith, 1996), distributed by the University of Hawaii (SOEST) and NOAA. Licence: LGPL.
      File used: gshhs_f.b (96 MB). Download with `download()`; it is not stored in git.

Lakes count as land here: they are not sea, and the model is not meant for them.
A safety margin is added around the coast, because pixels there mix land and sea and the coastline itself
can be off by a few hundred metres in the Arctic.

The margin is set with LAND_MARGIN_M (metres) as an environment variable or in the .env file in the repo root
(see .env.example). Without it the margin is 1000 m.
"""
from __future__ import annotations

import math
import os
import struct
import warnings
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
MARGIN_M = 1000.0            # default: pixels closer than this to land are also masked
MARGIN_ENV = "LAND_MARGIN_M"
MIN_SAFE_MARGIN_M = 400.0    # the model looks at a 720 m window: closer to land than half of it, land leaks into the features
LEVEL_LAND = 1               # GSHHG levels: 1 land, 2 lake, 3 island in lake, 4 pond, 5-6 Antarctica
HEADER = struct.Struct(">11i")   # id, n points, flag, west, east, south, north, area, area_full, container, ancestor


def margin_from_env(env_file: Path = ROOT / ".env") -> float:
    """Safety margin in metres: environment variable LAND_MARGIN_M, else the .env file, else MARGIN_M."""
    raw = os.environ.get(MARGIN_ENV)
    if raw is None and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(MARGIN_ENV + "="):
                raw = line.split("=", 1)[1].split("#")[0].strip().strip("'\"")
    if raw is None or not raw.strip():
        return MARGIN_M
    try:
        margin = float(raw)
    except ValueError:
        raise ValueError(f"{MARGIN_ENV} must be a number of metres, got {raw!r}") from None
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"{MARGIN_ENV} must be zero or more metres, got {raw!r}")
    if margin < MIN_SAFE_MARGIN_M:
        warnings.warn(f"{MARGIN_ENV}={margin:g} m is below {MIN_SAFE_MARGIN_M:g} m: ice / water answers next to the coast "
                      "are computed from pixels that contain land", stacklevel=2)
    return margin


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


def land_and_margin(x: np.ndarray, y: np.ndarray, epsg: int, margin_m: float | None = None,
                    path: Path = GSHHG_FILE) -> tuple[np.ndarray, np.ndarray]:
    """Two boolean grids (len(y), len(x)): real land, and the safety strip of sea within `margin_m` of land.

    `margin_m=None` takes the margin from LAND_MARGIN_M (see `margin_from_env`).
    Land just outside the grid still casts its margin into it.
    """
    if margin_m is None:
        margin_m = margin_from_env()
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing. Run: python -c 'from scripts.land_mask import download; download()'")
    polygons, bounds = _arctic_land(str(path), epsg)
    spacing = abs(float(x[1] - x[0]))
    pad = int(np.ceil(margin_m / spacing))                 # extra pixels on every side, cut off again below
    steps = np.arange(-pad, pad + 1)[:pad] * spacing       # -pad .. -1
    wide_x = np.concatenate([x[0] + steps, x, x[-1] - steps[::-1]])
    wide_y = np.concatenate([y[0] - steps, y, y[-1] + steps[::-1]])
    x0, x1, y0, y1 = wide_x.min() - spacing, wide_x.max() + spacing, wide_y.min() - spacing, wide_y.max() + spacing
    near = (bounds[:, 2] >= x0) & (bounds[:, 0] <= x1) & (bounds[:, 3] >= y0) & (bounds[:, 1] <= y1)
    land = rasterize_land([p for p, hit in zip(polygons, near) if hit], wide_x, wide_y)
    inner = (slice(pad, pad + len(y)), slice(pad, pad + len(x)))
    with_margin = add_margin(land, spacing, margin_m)[inner]
    land = land[inner]
    return land, with_margin & ~land


def land_mask(x: np.ndarray, y: np.ndarray, epsg: int, margin_m: float | None = None, path: Path = GSHHG_FILE) -> np.ndarray:
    """Boolean grid (len(y), len(x)): True = land or within the safety margin of land."""
    land, margin = land_and_margin(x, y, epsg, margin_m, path)
    return land | margin
