"""Put a Sentinel-1 GRD band on a map grid (e.g. the NISAR grid), calibrated to sigma0.

Sentinel-1 GRD images are in radar geometry (rows = flight direction, columns = distance from the
satellite). NISAR GCOV is on a map grid. To compare or track across sensors both must share one grid.

Steps: DN -> sigma0 (calibration table from the annotation XML) -> warp to the target grid using
the tie points (GCPs) stored in the image.
Thermal noise is NOT removed here. That is fine for HH over ice; use the notebook's noise step for HV.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


def href_to_path(href: str) -> str:
    return unquote(urlparse(href).path) if href.startswith("file://") else href


def calibration_table(xml_path: str):
    """Return (lines, pixels, sigmaNought values) from a calibration annotation file."""
    vectors = ET.parse(xml_path).getroot().find("calibrationVectorList")
    lines = np.array([int(v.find("line").text) for v in vectors])
    pixels = [np.array(v.find("pixel").text.split(), dtype="int64") for v in vectors]
    values = np.array([np.array(v.find("sigmaNought").text.split(), dtype="float64") for v in vectors])
    if any(not np.array_equal(p, pixels[0]) for p in pixels):
        raise ValueError("calibration vectors use different pixel positions")
    return lines, pixels[0], values


def interpolate_table(lines, pixels, values, n_rows: int, n_cols: int) -> np.ndarray:
    """Bilinear interpolation of the sparse table to every image pixel (float32, n_rows x n_cols)."""
    along_range = np.stack([np.interp(np.arange(n_cols), pixels, v) for v in values])  # (n_lines, n_cols)
    rows = np.arange(n_rows)
    upper = np.clip(np.searchsorted(lines, rows, side="right"), 1, len(lines) - 1)
    lower = upper - 1
    weight = np.clip((rows - lines[lower]) / (lines[upper] - lines[lower]), 0, 1)[:, None]
    return (along_range[lower] * (1 - weight) + along_range[upper] * weight).astype("float32")


def sigma0_radar_geometry(item, pol: str):
    """Calibrated sigma0 (linear power, NaN = no data) in radar geometry, plus the image's tie points."""
    with rasterio.open(href_to_path(item.assets[pol].href)) as src:
        dn = src.read(1).astype("float32")
        gcps, gcp_crs = src.gcps
    table = calibration_table(href_to_path(item.assets[f"schema-calibration-{pol}"].href))
    lut = interpolate_table(*table, *dn.shape)
    sigma0 = np.where(dn > 0, dn * dn / (lut * lut), np.nan).astype("float32")
    return sigma0, gcps, gcp_crs


def warp_to_grid(image, gcps, gcp_crs, x, y, epsg: int) -> np.ndarray:
    """Resample a radar-geometry image onto the grid with pixel centres x (ascending), y (descending)."""
    spacing = float(x[1] - x[0])
    if not np.isclose(y[0] - y[1], spacing):
        raise ValueError("target grid must have square pixels and descending y")
    # Convert tie points to the target projection first: avoids trouble at the 180 degree meridian.
    to_map = Transformer.from_crs(gcp_crs, CRS.from_epsg(epsg), always_xy=True)
    gx, gy = to_map.transform([g.x for g in gcps], [g.y for g in gcps])
    map_gcps = [GroundControlPoint(row=g.row, col=g.col, x=px, y=py) for g, px, py in zip(gcps, gx, gy)]
    out = np.full((len(y), len(x)), np.nan, dtype="float32")
    reproject(image, out, gcps=map_gcps, src_crs=CRS.from_epsg(epsg), src_nodata=np.nan,
              dst_transform=from_origin(x[0] - spacing / 2, y[0] + spacing / 2, spacing, spacing),
              dst_crs=CRS.from_epsg(epsg), dst_nodata=np.nan, resampling=Resampling.average)
    return out
