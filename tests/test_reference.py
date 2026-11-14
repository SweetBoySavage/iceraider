import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reference import LAND, NISE_SIZE, WATER, hdf4_blocks, read_nise_north, sample, score_water_mask, summarise

N = NISE_SIZE * NISE_SIZE


def fake_hdf4(grids):
    """Minimal HDF4 file: magic, one directory block, then the grids."""
    header = 4 + 6 + 12 * len(grids)
    out = b"\x0e\x03\x13\x01" + struct.pack(">HI", len(grids), 0)
    for i, g in enumerate(grids):
        out += struct.pack(">HHII", 702, i + 1, header + sum(len(x) for x in grids[:i]), len(g))
    return out + b"".join(grids)


def grid(fill, ice_value=None):
    a = np.full((NISE_SIZE, NISE_SIZE), fill, np.uint8)
    if ice_value:
        a[360, 360] = ice_value
        a[0, 0] = 254
        a[10, 10] = 0
    return a.tobytes()


def test_hdf4_blocks_found_in_file_order_and_bad_file_rejected():
    data = fake_hdf4([b"a" * 5, b"b" * 7, b"c" * 5])
    assert hdf4_blocks(data, 5) == [b"a" * 5, b"c" * 5]
    with pytest.raises(ValueError):
        hdf4_blocks(b"nope", 5)


def test_nise_values_become_ice_water_land(tmp_path):
    f = tmp_path / "n.HDFEOS"
    f.write_bytes(fake_hdf4([grid(255), grid(255, ice_value=80), grid(255), grid(255)]))
    conc, transform, crs = read_nise_north(f)
    assert conc[360, 360] == 80 and conc[5, 5] == WATER and conc[10, 10] == LAND and np.isnan(conc[0, 0])
    assert transform * (360.5, 360.5) == pytest.approx((0, 0), abs=1)      # the pole is the grid centre


def test_nise_day_without_ice_fails_fast(tmp_path):
    f = tmp_path / "n.HDFEOS"
    f.write_bytes(fake_hdf4([grid(255)] * 4))
    with pytest.raises(ValueError, match="data gap"):
        read_nise_north(f)


def test_sample_and_summarise():
    conc = np.array([[100, 100], [0, LAND]], "float32")                    # 10 km cells, top-left at (0, 20 km)
    ref = sample(conc, from_origin(0, 20000, 10000, 10000), "EPSG:3413", np.array([5000.0, 15000.0]),
                 np.array([15000.0, 5000.0]), 3413)
    assert ref.tolist() == [[100, 100], [0, LAND]]
    s = summarise(ref)
    assert s == {"land_pct": 25.0, "open_water_pct": 33.3, "ice_concentration_mean": 66.7, "ice_concentration_min": 0.0}
    outside = sample(conc, from_origin(0, 20000, 10000, 10000), "EPSG:3413", np.array([-5000.0]), np.array([5000.0]), 3413)
    assert np.isnan(outside).all()


def test_score_counts_only_clear_reference_cells():
    ref = np.array([0.0, 5.0, 50.0, 90.0, 100.0, LAND])
    pred = np.array([True, False, True, False, False, True])
    s = score_water_mask(pred, ref, np.ones(6, bool))
    assert s["reference_water_px"] == 2 and s["reference_ice_px"] == 2
    assert s["water_found_pct"] == 50.0 and s["ice_found_pct"] == 100.0 and s["balanced_accuracy_pct"] == 75.0
    only_ice = score_water_mask(np.array([False, True]), np.array([100.0, 95.0]), np.ones(2, bool))
    assert only_ice["water_found_pct"] is None and only_ice["balanced_accuracy_pct"] == 50.0
    with pytest.raises(ValueError):
        score_water_mask(np.array([True]), np.array([50.0]), np.ones(1, bool))


def test_certain_labels_drop_the_edge_and_land():
    from scripts.reference import ICE, OPEN_WATER, UNSURE, certain_labels
    conc = np.full((7, 9), 100.0)
    conc[:, :4] = 0.0            # water on the left, ice on the right
    conc[0, 8] = LAND
    lab = certain_labels(conc)
    assert lab[3, 0] == OPEN_WATER and lab[3, 2] == OPEN_WATER
    assert lab[3, 3] == UNSURE and lab[3, 4] == UNSURE          # cells touching the edge
    assert lab[3, 5] == ICE and lab[5, 8] == ICE
    assert lab[0, 8] == UNSURE and lab[1, 7] == UNSURE          # land and its neighbours
    assert (certain_labels(conc, margin_cells=0)[3, 3:5] == [OPEN_WATER, ICE]).all()


def test_points_to_mask_marks_the_right_pixel():
    from pyproj import Transformer
    from scripts.reference import points_to_mask
    x, y = np.arange(-1_000_000, -999_000, 80.0), np.arange(500_000, 499_000, -80.0)
    lon, lat = Transformer.from_crs(3413, 4326, always_xy=True).transform([x[3] + 10, 0.0], [y[5] - 10, 0.0])
    mask = points_to_mask(np.array(lon), np.array(lat), x, y, 3413)
    assert mask.sum() == 1 and mask[5, 3]


def test_icesat2_keeps_only_clear_good_ice_segments(tmp_path):
    import h5py
    from scripts.reference import read_icesat2_ice
    p = tmp_path / "atl07.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("gt1l/sea_ice_segments")
        g["longitude"], g["latitude"] = np.arange(5.0), np.arange(70.0, 75.0)
        h, st = g.create_group("heights"), g.create_group("stats")
        h["height_segment_type"] = np.array([1, 1, 2, 1, 1], "int8")              # index 2 is a lead
        h["height_segment_quality"] = np.array([1, 0, 1, 1, 1], "int8")           # index 1 is bad quality
        h["height_segment_fit_quality_flag"] = np.array([1, 1, 1, 5, 2], "int8")  # index 3 is a poor fit
        st["cloud_flag_asr"] = np.array([0, 0, 0, 0, 4], "int8")                  # index 4 is cloudy
        st["layer_flag"] = np.zeros(5, "int8")
    lon, lat = read_icesat2_ice(p)
    assert lon.tolist() == [0.0] and lat.tolist() == [70.0]
