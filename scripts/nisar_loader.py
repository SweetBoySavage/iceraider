"""Load a NISAR L2 GCOV scene (HDF5) into the arrays the notebook's filter + classify steps expect.

GCOV is already calibrated and geocoded, so the notebook's calibration and noise cells are skipped.
What the files contain (checked on two provisional sea-ice scenes, Oct 2026):
  - values are gamma0 in LINEAR POWER (not dB, not amplitude)
  - thermal noise is NOT subtracted; a coarse noise map (NESZ) is stored next to the data
  - each polarisation lives in its own frequency group (e.g. VV in frequencyA, HH in frequencyB)
  - sea-ice scenes use a polar stereographic grid (EPSG:3413) with 80 m pixels

Usage:
    from scripts.nisar_loader import load_gcov
    scene = load_gcov(path, window=(row0, col0, size))
    scene.power["hh"]   # 2-D float array, gamma0 linear power, NaN = no data
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

GCOV = "science/LSAR/GCOV"


@dataclass
class GcovScene:
    name: str
    power: dict[str, np.ndarray]   # per polarisation ("hh", "vv", ...): gamma0, linear power
    nesz: dict[str, np.ndarray]    # per polarisation: noise-equivalent backscatter, same grid, linear power
    looks: dict[str, float]        # per polarisation: median number of looks in the window
    x: np.ndarray                  # projected x of pixel centres (m)
    y: np.ndarray                  # projected y of pixel centres (m)
    epsg: int
    spacing_m: float
    noise_removed: bool
    incidence: np.ndarray | None = None   # viewing angle from vertical, degrees, same grid (None if not in file)


def list_polarisations(f: h5py.File) -> dict[str, str]:
    """Map polarisation (lower case) -> frequency group name, e.g. {"vv": "frequencyA", "hh": "frequencyB"}."""
    found = {}
    for freq, group in f[f"{GCOV}/grids"].items():
        if not freq.startswith("frequency"):
            continue
        for pol in group["listOfPolarizations"][:]:
            pol = pol.decode().lower()
            if pol in found:
                raise ValueError(f"{pol.upper()} is in both {found[pol]} and {freq}; choose one explicitly")
            found[pol] = freq
    return found


def fill_nan_with_nearest(values: np.ndarray) -> np.ndarray:
    """Replace NaN cells by the nearest valid cell. The noise map is empty outside the footprint,
    and without this the image pixels next to the footprint edge would become NaN too."""
    empty = np.isnan(values)
    if empty.all():
        raise ValueError("noise map has no valid values")
    if not empty.any():
        return values
    nearest = distance_transform_edt(empty, return_distances=False, return_indices=True)
    return values[tuple(nearest)]


def _noise_on_grid(f: h5py.File, freq: str, pol: str, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Interpolate the coarse noise map onto the image grid."""
    n = f[f"{GCOV}/metadata/calibrationInformation/{freq}/noiseEquivalentBackscatter"]
    ny, nx, values = n["yCoordinates"][:], n["xCoordinates"][:], n[pol.upper()][:]
    if ny[0] > ny[-1]:                       # interpolator needs ascending axes
        ny, values = ny[::-1], values[::-1]
    interp = RegularGridInterpolator((ny, nx), fill_nan_with_nearest(values), bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return interp((yy, xx)).astype("float32")


def _incidence_on_grid(f: h5py.File, y: np.ndarray, x: np.ndarray):
    """Incidence angle at sea level, interpolated from the coarse radar-geometry cube."""
    name = f"{GCOV}/metadata/radarGrid"
    if name not in f or "incidenceAngle" not in f[name]:
        return None
    g = f[name]
    level = int(np.argmin(np.abs(g["heightAboveEllipsoid"][:])))
    gy, gx, values = g["yCoordinates"][:], g["xCoordinates"][:], g["incidenceAngle"][level]
    if gy[0] > gy[-1]:
        gy, values = gy[::-1], values[::-1]
    interp = RegularGridInterpolator((gy, gx), fill_nan_with_nearest(values), bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return interp((yy, xx)).astype("float32")


def load_gcov(path, pols=None, window=None, subtract_noise=True) -> GcovScene:
    """Read backscatter for `pols` (default: all in the file).

    window: (row0, col0, size) in pixels, or None for the full scene.
    subtract_noise: subtract the noise map from the power. Results may be <= 0 in very dark areas;
                    keep them for filtering and clip only when converting to dB (as the notebook does).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        available = list_polarisations(f)
        pols = list(pols) if pols else sorted(available)
        missing = [p for p in pols if p not in available]
        if missing:
            raise KeyError(f"{missing} not in file; available: {sorted(available)}")
        applied = f[f"{GCOV}/metadata/processingInformation/parameters/noiseCorrectionApplied"][()].decode()
        already_removed = applied.lower() == "true"

        power, nesz, looks, grids = {}, {}, {}, []
        for pol in pols:
            g = f[f"{GCOV}/grids/{available[pol]}"]
            n_rows, n_cols = g[pol.upper() * 2].shape
            r0, c0, size = window if window else (0, 0, None)
            rows = slice(r0, r0 + size if size else n_rows)
            cols = slice(c0, c0 + size if size else n_cols)
            x, y = g["xCoordinates"][cols], g["yCoordinates"][rows]
            data = g[pol.upper() * 2][rows, cols].astype("float64")
            if data.size == 0:
                raise ValueError(f"window {window} is outside the scene ({n_rows} x {n_cols})")
            nesz[pol] = _noise_on_grid(f, available[pol], pol, y, x)
            if subtract_noise and not already_removed:
                data = data - nesz[pol]
            power[pol] = data
            looks[pol] = float(np.nanmedian(g["numberOfLooks"][rows, cols]))
            grids.append((x, y, int(g["projection"][()]), float(g["xCoordinateSpacing"][()])))

        x, y, epsg, spacing = grids[0]
        for gx, gy, gepsg, _ in grids[1:]:
            if gepsg != epsg or gx.shape != x.shape or not (np.allclose(gx, x) and np.allclose(gy, y)):
                raise ValueError("polarisations are on different grids; load them separately")
        incidence = _incidence_on_grid(f, y, x)

    return GcovScene(name=path.stem, power=power, nesz=nesz, looks=looks, x=x, y=y, epsg=epsg,
                     spacing_m=spacing, noise_removed=already_removed or subtract_noise, incidence=incidence)
