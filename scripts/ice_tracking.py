"""Track sea-ice motion between two images on the same map grid, with phase cross-correlation.

Idea: cut both images into square patches. For each patch, find the shift that best lines up
image 1 with image 2. Shift x pixel size / time gap = ice velocity.

Two passes, because ice can move further than one small patch is wide:
  1. coarse: shrunk images, big patches -> rough shift (catches large moves)
  2. fine:   full-resolution patches, image 2 read at the rough shift -> exact shift

Bad matches are dropped with a similarity score (0 = unrelated, 1 = identical) checked after shifting.
Then keep_on_ice() drops arrows that start in water (ice mask of image 1) or end in water (image 2).

Full description: docs/ice_tracking.md
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation


def highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Remove slow brightness changes (e.g. different viewing angle) so only texture is matched."""
    return img - gaussian_filter(img, sigma)


def shrink(img: np.ndarray, n: int) -> np.ndarray:
    h, w = (img.shape[0] // n) * n, (img.shape[1] // n) * n
    return img[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation of two equal-size patches."""
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _patch(img: np.ndarray, row: int, col: int, size: int):
    """Patch with top-left corner (row, col), or None if it leaves the image or has no-data."""
    if row < 0 or col < 0 or row + size > img.shape[0] or col + size > img.shape[1]:
        return None
    p = img[row:row + size, col:col + size]
    return p if np.isfinite(p).all() else None


def _shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(d_row, d_col) that the content of `a` moved to reach its place in `b`."""
    shift, _, _ = phase_cross_correlation(b, a, upsample_factor=10, normalization=None)
    return shift


def track(img1, img2, patch=128, step=64, coarse=4, min_similarity=0.5):
    """Displacement of features from img1 to img2, in pixels.

    Returns a dict of 1-D arrays: row, col (patch centre in img1), d_row, d_col, similarity.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"images must share one grid; got {img1.shape} and {img2.shape}")
    small1, small2 = shrink(img1, coarse), shrink(img2, coarse)
    out = {k: [] for k in ("row", "col", "d_row", "d_col", "similarity")}
    for row in range(0, img1.shape[0] - patch + 1, step):
        for col in range(0, img1.shape[1] - patch + 1, step):
            a = _patch(img1, row, col, patch)
            if a is None:
                continue
            # pass 1: a big patch (same pixel count, `coarse` times wider) centred on the same spot
            centre_r, centre_c = (row + patch // 2) // coarse, (col + patch // 2) // coarse
            r0, c0 = centre_r - patch // 2, centre_c - patch // 2
            big1, big2 = _patch(small1, r0, c0, patch), _patch(small2, r0, c0, patch)
            if big1 is None or big2 is None:
                continue
            rough = np.round(_shift(big1, big2) * coarse).astype(int)
            # pass 2: full resolution, image 2 read at the rough shift
            b = _patch(img2, row + rough[0], col + rough[1], patch)
            if b is None:
                continue
            fine = _shift(a, b)
            total = rough + fine
            aligned = _patch(img2, row + int(round(total[0])), col + int(round(total[1])), patch)
            if aligned is None:
                continue
            score = similarity(a, aligned)
            if score < min_similarity:
                continue
            for key, value in zip(out, (row + patch // 2, col + patch // 2, total[0], total[1], score)):
                out[key].append(value)
    return {k: np.array(v, dtype="float64") for k, v in out.items()}


def keep_on_ice(tracks: dict, ice1: np.ndarray, ice2: np.ndarray) -> tuple[dict, np.ndarray]:
    """Drop arrows that start in water (ice mask of image 1) or end in water (ice mask of image 2).

    ice1, ice2: bool arrays on the tracking grid, True = ice (from an ice/water detection run on each
    image separately). An arrow starts at its patch centre (row, col) and ends at
    (row + d_row, col + d_col), rounded to the nearest pixel. An end outside the grid counts as water.

    Returns (kept tracks, bool array `keep` over the input arrows).
    """
    if ice1.shape != ice2.shape:
        raise ValueError(f"ice masks must share one grid; got {ice1.shape} and {ice2.shape}")
    r0, c0 = tracks["row"].astype(int), tracks["col"].astype(int)
    r1, c1 = np.round(tracks["row"] + tracks["d_row"]).astype(int), np.round(tracks["col"] + tracks["d_col"]).astype(int)
    h, w = ice2.shape
    inside = (r1 >= 0) & (r1 < h) & (c1 >= 0) & (c1 < w)
    keep = ice1[r0, c0] & inside
    keep[inside] &= ice2[r1[inside], c1[inside]]
    kept = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape else v) for k, v in tracks.items()}
    return kept, keep


def to_velocity(d_row, d_col, spacing_m: float, hours: float, y_down: bool = True):
    """Pixel shifts -> (vx, vy) in m/s on the map grid, and speed in km/day."""
    seconds = hours * 3600
    vx = d_col * spacing_m / seconds
    vy = d_row * spacing_m / seconds * (-1 if y_down else 1)
    speed_km_day = np.hypot(vx, vy) * 86.4
    return vx, vy, speed_km_day
