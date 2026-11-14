"""Independent reference maps of sea ice concentration (percent of each cell covered by ice).

Both come from the AMSR2 passive microwave sensor, which sees through cloud and darkness but is coarse:
  - Uni Bremen ASI, 3.125 km grid, GeoTIFF, EPSG:3413. 0 = open water / no data, 1-100 = % ice, 120 = land.
  - NASA NSIDC NISE_A2, 25 km EASE-Grid, old HDF4 format. 1-100 = % ice, 255 = ocean, 0/101-103/252 = land or coast.
    (NASA's AU_SI12 stops on 31 Aug 2025, and NISE_A2 has no ice data for 24-28 Oct 2026.)
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

BREMEN_URL = ("https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/{y}/{mon}/Arctic3125/"
              "asi-AMSR2-n3125-{y}{m}{d}-v5.4.tif")
NISE_URL = ("https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/OTHR/NISE_A2/1/"
            "{y}/{m}/{d}/NISE_AMSR2_{y}{m}{d}.HDFEOS")
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
LAND, WATER = -1.0, 0.0            # codes used in the arrays this module returns; ice is 1..100
NISE_SIZE, NISE_CELL_M, NISE_EPSG = 721, 25067.525, 3408


def download(date: str, kind: str, folder: Path) -> Path:
    """date = 'YYYYMMDD', kind = 'bremen' or 'nise'. NISE needs the Earthdata login in ~/.netrc."""
    import requests
    y, m, d = date[:4], date[4:6], date[6:]
    url = (BREMEN_URL if kind == "bremen" else NISE_URL).format(y=y, m=m, d=d, mon=MONTHS[int(m) - 1])
    path = Path(folder) / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.Session().get(url, timeout=300)       # the session reads ~/.netrc on the login redirect
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def read_bremen(path):
    """Return (concentration, transform, crs): float array with LAND / WATER / 1..100."""
    with rasterio.open(path) as src:
        raw, transform, crs = src.read(1), src.transform, src.crs
    conc = raw.astype("float32")
    conc[raw > 100] = LAND
    return conc, transform, crs


def hdf4_blocks(data: bytes, length: int) -> list[bytes]:
    """All data blocks of exactly `length` bytes in an HDF4 file, in file order (no HDF4 library needed)."""
    if data[:4] != b"\x0e\x03\x13\x01":
        raise ValueError("not an HDF4 file")
    blocks, pos = [], 4
    while pos:
        count, next_pos = struct.unpack(">HI", data[pos:pos + 6])
        for i in range(count):
            _tag, _ref, offset, size = struct.unpack(">HHII", data[pos + 6 + 12 * i:pos + 18 + 12 * i])
            if size == length:
                blocks.append((offset, data[offset:offset + size]))
        pos = next_pos
    return [b for _, b in sorted(blocks)]


def read_nise_north(path):
    """Return (concentration, transform, crs) for the northern grid, same codes as read_bremen."""
    blocks = hdf4_blocks(Path(path).read_bytes(), NISE_SIZE * NISE_SIZE)
    if len(blocks) != 4:
        raise ValueError(f"expected 4 grids (north/south, age/extent), found {len(blocks)}")
    raw = np.frombuffer(blocks[1], np.uint8).reshape(NISE_SIZE, NISE_SIZE)       # order: N age, N extent, S age, S extent
    is_ice = (raw >= 1) & (raw <= 100)
    if not is_ice.any():
        raise ValueError(f"{Path(path).name} has no sea-ice values (data gap on that day)")
    conc = np.full(raw.shape, LAND, "float32")
    conc[raw == 255] = WATER
    conc[is_ice] = raw[is_ice]
    conc[raw == 254] = np.nan                                                    # grid corners, off the Earth
    half = NISE_SIZE * NISE_CELL_M / 2
    return conc, rasterio.transform.from_origin(-half, half, NISE_CELL_M, NISE_CELL_M), f"EPSG:{NISE_EPSG}"


def sample(conc, transform, crs, x, y, epsg: int):
    """Reference value at each point of the map grid x (1-D), y (1-D) in `epsg`. Returns (len(y), len(x))."""
    xx, yy = np.meshgrid(x, y)
    px, py = Transformer.from_crs(epsg, crs, always_xy=True).transform(xx, yy)
    cols, rows = (~transform) * (px, py)
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    inside = (rows >= 0) & (rows < conc.shape[0]) & (cols >= 0) & (cols < conc.shape[1])
    out = np.full(xx.shape, np.nan, "float32")
    out[inside] = conc[rows[inside], cols[inside]]
    return out


def summarise(ref: np.ndarray) -> dict:
    """Shares of a sampled reference area, in percent."""
    known = np.isfinite(ref)
    if not known.any():
        raise ValueError("reference has no data in this area")
    sea = known & (ref != LAND)
    return {"land_pct": round(100 * float(np.mean(ref[known] == LAND)), 1),
            "open_water_pct": round(100 * float(np.mean(ref[sea] < 15)), 1) if sea.any() else None,
            "ice_concentration_mean": round(float(ref[sea].mean()), 1) if sea.any() else None,
            "ice_concentration_min": round(float(ref[sea].min()), 1) if sea.any() else None}


def score_water_mask(is_water: np.ndarray, ref: np.ndarray, valid: np.ndarray, water_below=15, ice_from=80) -> dict:
    """Compare a water mask with a sampled reference. Only clear reference cells count:
    concentration < `water_below` % = water, >= `ice_from` % = ice. The mixed zone in between is skipped."""
    ref_water = valid & (ref >= 0) & (ref < water_below)
    ref_ice = valid & (ref >= ice_from)
    found_water = float(np.mean(is_water[ref_water])) if ref_water.any() else None
    found_ice = float(np.mean(~is_water[ref_ice])) if ref_ice.any() else None
    both = [v for v in (found_water, found_ice) if v is not None]
    if not both:
        raise ValueError("reference has no clear water or ice in this area")
    return {"reference_water_px": int(ref_water.sum()), "reference_ice_px": int(ref_ice.sum()),
            "water_found_pct": None if found_water is None else round(100 * found_water, 1),
            "ice_found_pct": None if found_ice is None else round(100 * found_ice, 1),
            "balanced_accuracy_pct": round(100 * float(np.mean(both)), 1)}


# ---------------------------------------------------------------------------------------------
# Labels for training and testing: combine the references and keep only what is certain.
# ---------------------------------------------------------------------------------------------
ICE, OPEN_WATER, UNSURE = 1, 0, -1


def certain_labels(conc: np.ndarray, water_below=15, ice_from=80, margin_cells=1) -> np.ndarray:
    """Per reference cell: ICE, OPEN_WATER or UNSURE.

    A cell counts only if it AND all neighbours within `margin_cells` agree (all >= ice_from, or all
    < water_below, and no land). This drops the blurry ice edge, where a 3 km cell is a mix.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    size = 2 * margin_cells + 1
    known = np.isfinite(conc) & (conc != LAND)
    filled_low, filled_high = np.where(known, conc, -np.inf), np.where(known, conc, np.inf)
    lowest, highest = minimum_filter(filled_low, size, mode="nearest"), maximum_filter(filled_high, size, mode="nearest")
    out = np.full(conc.shape, UNSURE, "int8")
    out[known & (lowest >= ice_from)] = ICE
    out[known & (highest < water_below)] = OPEN_WATER
    return out


def read_icesat2_ice(path, max_cloud_flag=1):
    """Lon, lat of ICESat-2 ATL07 segments that are certainly ice or snow surface (all six beams).

    Kept only if: surface type 'other' (not cloud, not a lead), good quality, good height fit, and the
    atmosphere flags say clear. ATL07 has no data over open ocean, so it cannot confirm open water.
    """
    import h5py
    lons, lats = [], []
    with h5py.File(path, "r") as f:
        for beam in ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"):
            if beam not in f or "sea_ice_segments" not in f[beam]:
                continue
            g = f[f"{beam}/sea_ice_segments"]
            h, st = g["heights"], g["stats"]
            fit = h["height_segment_fit_quality_flag"][:]
            good = ((h["height_segment_type"][:] == 1) & (h["height_segment_quality"][:] == 1) & (fit >= 1) & (fit <= 3)
                    & (st["cloud_flag_asr"][:] <= max_cloud_flag) & (st["layer_flag"][:] == 0))
            lons.append(g["longitude"][:][good])
            lats.append(g["latitude"][:][good])
    if not lons:
        raise ValueError(f"no sea-ice beams in {path}")
    return np.concatenate(lons), np.concatenate(lats)


def points_to_mask(lon, lat, x, y, epsg: int) -> np.ndarray:
    """Boolean grid (len(y), len(x)): True in pixels that contain at least one point."""
    px, py = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    spacing = float(x[1] - x[0])
    cols, rows = np.round((px - x[0]) / spacing).astype(int), np.round((y[0] - py) / spacing).astype(int)
    inside = (rows >= 0) & (rows < len(y)) & (cols >= 0) & (cols < len(x))
    mask = np.zeros((len(y), len(x)), bool)
    mask[rows[inside], cols[inside]] = True
    return mask
