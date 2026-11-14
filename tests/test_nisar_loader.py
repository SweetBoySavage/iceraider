"""Tests for scripts/nisar_loader.py on a tiny synthetic GCOV file (same layout as the real product)."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nisar_loader import GCOV, fill_nan_with_nearest, list_polarisations, load_gcov

X = np.arange(0, 800, 80.0)            # 10 columns
Y = np.arange(1600, 0, -80.0)          # 20 rows, descending like the real files
NOISE = 0.001


def make_file(path, noise_applied="False"):
    with h5py.File(path, "w") as f:
        for freq, pol, value in (("frequencyA", "VV", 0.02), ("frequencyB", "HH", 0.03)):
            g = f.create_group(f"{GCOV}/grids/{freq}")
            data = np.full((Y.size, X.size), value, "float32")
            data[0, 0] = np.nan
            g[pol * 2] = data
            g["listOfPolarizations"] = np.array([pol.encode()])
            g["xCoordinates"], g["yCoordinates"] = X, Y
            g["numberOfLooks"] = np.full(data.shape, 39.0, "float32")
            g["projection"] = np.uint32(3413)
            g["xCoordinateSpacing"] = 80.0
            n = f.create_group(f"{GCOV}/metadata/calibrationInformation/{freq}/noiseEquivalentBackscatter")
            n["xCoordinates"], n["yCoordinates"] = X[::3], Y[::3]
            n[pol] = np.full((Y[::3].size, X[::3].size), NOISE, "float32")
        f[f"{GCOV}/metadata/processingInformation/parameters/noiseCorrectionApplied"] = noise_applied.encode()
    return path


def test_finds_each_polarisation_in_its_frequency_group(tmp_path):
    with h5py.File(make_file(tmp_path / "a.h5")) as f:
        assert list_polarisations(f) == {"vv": "frequencyA", "hh": "frequencyB"}


def test_loads_power_and_subtracts_noise(tmp_path):
    s = load_gcov(make_file(tmp_path / "a.h5"))
    assert s.power["hh"].shape == (20, 10) and s.epsg == 3413 and s.spacing_m == 80
    assert s.power["hh"][5, 5] == pytest.approx(0.03 - NOISE, rel=1e-4)
    assert s.power["vv"][5, 5] == pytest.approx(0.02 - NOISE, rel=1e-4)
    assert np.isnan(s.power["hh"][0, 0])
    assert s.looks["hh"] == 39 and s.noise_removed


def test_noise_is_not_subtracted_twice(tmp_path):
    s = load_gcov(make_file(tmp_path / "a.h5", noise_applied="True"))
    assert s.power["hh"][5, 5] == pytest.approx(0.03, rel=1e-4)


def test_can_skip_noise_subtraction(tmp_path):
    s = load_gcov(make_file(tmp_path / "a.h5"), subtract_noise=False)
    assert s.power["hh"][5, 5] == pytest.approx(0.03, rel=1e-4) and not s.noise_removed


def test_window_cuts_data_and_coordinates(tmp_path):
    s = load_gcov(make_file(tmp_path / "a.h5"), pols=["hh"], window=(2, 3, 4))
    assert s.power["hh"].shape == (4, 4) and list(s.power) == ["hh"]
    assert s.x[0] == X[3] and s.y[0] == Y[2]


def test_unknown_polarisation_fails_fast(tmp_path):
    with pytest.raises(KeyError):
        load_gcov(make_file(tmp_path / "a.h5"), pols=["hv"])


def test_empty_noise_cells_take_the_nearest_value():
    grid = np.array([[np.nan, 1.0, np.nan, np.nan, 5.0]])
    assert fill_nan_with_nearest(grid).tolist() == [[1.0, 1.0, 1.0, 5.0, 5.0]]


def test_all_empty_noise_map_fails_fast():
    with pytest.raises(ValueError):
        fill_nan_with_nearest(np.full((2, 2), np.nan))
