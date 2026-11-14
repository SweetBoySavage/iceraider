"""Speckle filter and brightness classes. Same methods as notebooks/sea_ice_sar.ipynb, importable."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans

POWER_FLOOR = 1e-4  # -40 dB: clip before taking the logarithm


def to_db(power: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, POWER_FLOOR))


def lee_filter(img: np.ndarray, enl: float, size: int = 7) -> np.ndarray:
    """Lee speckle filter on linear power. `enl` = number of looks of the input."""
    mean = uniform_filter(img, size)
    var = np.maximum(uniform_filter(img * img, size) - mean**2, 0)
    weight = np.clip(1 - (mean**2 / enl) / np.maximum(var, 1e-12), 0, 1)
    return mean + weight * (img - mean)


def kmeans_classes(db_a: np.ndarray, db_b: np.ndarray, k: int = 4):
    """k-means on two dB channels. Labels are ordered so 0 = darkest in channel a. Returns (labels, centres)."""
    X = np.column_stack([db_a.ravel(), db_b.ravel()])
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X[::7])
    order = np.argsort(km.cluster_centers_[:, 0])
    return np.argsort(order)[km.predict(X)].reshape(db_a.shape), km.cluster_centers_[order]


def dark_threshold(centres: np.ndarray) -> float:
    """Threshold on channel a: halfway between the two darkest k-means centres."""
    return float((centres[0, 0] + centres[1, 0]) / 2)


def largest_clean_square(valid: np.ndarray, sizes=(2000, 1750, 1500, 1250, 1000), step: int = 50):
    """(row, col, size) of the first square window without no-data, trying big sizes first."""
    for size in sizes:
        for r in range(0, valid.shape[0] - size, step):
            for c in range(0, valid.shape[1] - size, step):
                if valid[r:r + size:25, c:c + size:25].all() and valid[r:r + size, c:c + size].all():
                    return r, c, size
    raise ValueError("no clean square window found")
