import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_mosaic as bm
from scripts.evaluate_detection import NO_DATA, OPEN_WATER, SEA_ICE


def test_grid_holds_every_scene_plus_the_border_on_whole_cells():
    grid = bm.grid_around([(1010.0, 1990.0, -990.0, -10.0), (2500.0, 3100.0, 100.0, 450.0)], cell_m=400.0, pad_m=1000.0)
    assert (grid.x0, grid.y0) == (0.0, 1600.0)
    assert (grid.nx, grid.ny) == (11, 9)                     # x: 0 .. 4400, y: 1600 .. -2000
    assert grid.x[0] == 200.0 and grid.y[0] == 1400.0        # cell centres
    assert np.allclose(np.diff(grid.x), 400.0) and np.allclose(np.diff(grid.y), -400.0)


def test_grid_needs_at_least_one_scene():
    with pytest.raises(ValueError):
        bm.grid_around([])


def test_bin_scene_counts_ice_and_usable_pixels_per_cell_and_drops_pixels_outside():
    grid = bm.Grid(x0=0.0, y0=800.0, nx=2, ny=2, cell_m=400.0)
    x = np.arange(-40.0, 800.0, 80.0)                        # first column lies left of the grid
    y = np.arange(760.0, 0.0, -80.0)                         # 10 rows, 5 per cell
    ice = np.zeros((y.size, x.size), bool)
    ice[:5, :] = True                                        # top half is ice
    usable = np.ones_like(ice)
    usable[:, -1] = False                                    # last column unusable
    ice_n, seen = bm.bin_scene(ice, usable, x, y, grid)
    assert seen.tolist() == [[25, 20], [25, 20]]             # the column outside the grid is not counted
    assert ice_n.tolist() == [[25, 20], [0, 0]]


def test_cell_is_ice_from_a_quarter_ice_and_no_data_when_mostly_unseen():
    seen = np.array([[25, 25, 25, 12, 13]])
    ice_n = np.array([[0, 6, 7, 12, 0]])
    out = bm.cell_classes(ice_n, seen, pixels_per_cell=25, min_ice=0.25, min_seen=0.5)
    assert out.tolist() == [[OPEN_WATER, OPEN_WATER, SEA_ICE, NO_DATA, OPEN_WATER]]


def test_newer_scene_covers_older_one_but_its_gaps_do_not():
    classes = np.full((1, 3), NO_DATA, "uint8")
    index = np.full((1, 3), bm.NO_SCENE, "uint8")
    bm.paint(classes, index, np.array([[SEA_ICE, SEA_ICE, NO_DATA]], "uint8"), 0)
    bm.paint(classes, index, np.array([[NO_DATA, OPEN_WATER, OPEN_WATER]], "uint8"), 1)
    assert classes.tolist() == [[SEA_ICE, OPEN_WATER, OPEN_WATER]]
    assert index.tolist() == [[0, 1, 1]]


def test_ice_distance_is_in_metres_zero_on_ice_and_capped():
    classes = np.full((1, 6), OPEN_WATER, "uint8")
    classes[0, 0] = SEA_ICE
    classes[0, 2] = NO_DATA                                  # unknown cells are not ice
    assert bm.ice_distance_m(classes, 400.0, cap_m=1500).tolist() == [[0, 400, 800, 1200, 1500, 1500]]
    assert bm.ice_distance_m(np.full((2, 2), OPEN_WATER, "uint8"), 400.0, cap_m=1500).tolist() == [[1500, 1500]] * 2


def test_scenes_between_takes_every_downloaded_scene_of_the_period_in_time_order(tmp_path):
    def name(stamp):
        return f"NISAR_L2_PR_GCOV_034_045_D_051_0505_SVSH_A_{stamp}_{stamp}_P05036_N_F_J_001"
    for stamp in ("20261028T084333", "20261019T081800", "20261027T074517"):
        (tmp_path / f"{name(stamp)}.h5").touch()
    (tmp_path / "S1C_EW_20261027.npy").touch()                 # other files are ignored
    assert bm.scenes_between("20261027", "20261031", tmp_path) == [name("20261027T074517"), name("20261028T084333")]
    with pytest.raises(ValueError):
        bm.scenes_between("20250101", "20250102", tmp_path)


def test_mosaic_survives_saving_and_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "MAPS", tmp_path)
    mosaic = {"classes": np.array([[SEA_ICE, NO_DATA]], "uint8"), "scene_index": np.array([[0, bm.NO_SCENE]], "uint8"),
              "ice_distance_m": np.array([[0, 400]], "uint16"), "x": np.array([200.0, 600.0]), "y": np.array([-200.0]),
              "epsg": 3413, "cell_m": 400.0, "margin_m": 1000.0, "scenes": ["a"], "skipped": []}
    bm.save_mosaic(mosaic, bm.mosaic_path("20261027", "20261031"))
    back = bm.load_mosaic("20261027", "20261031")
    assert back["classes"].tolist() == [[SEA_ICE, NO_DATA]] and back["scenes"] == ["a"] and back["skipped"] == []
    assert back["cell_m"] == 400.0 and back["epsg"] == 3413
    with pytest.raises(FileNotFoundError):
        bm.load_mosaic("20250101", "20250102")
