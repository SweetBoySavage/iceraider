import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import route_safety as rs
from scripts.evaluate_detection import COAST_MARGIN, LAND, NO_DATA, OPEN_WATER, SEA_ICE


def bundle():
    """One row of ten 400 m cells: land, coast strip, 5 water, ice, 2 no data. Distances as the build makes them."""
    classes = np.array([[LAND, COAST_MARGIN, OPEN_WATER, OPEN_WATER, OPEN_WATER, OPEN_WATER, OPEN_WATER, SEA_ICE, NO_DATA, NO_DATA]], "uint8")
    cells = np.arange(10)
    return {"classes": classes, "scene_index": np.array([[255, 0, 0, 0, 0, 0, 0, 0, 255, 255]], "uint8"),
            "ice_distance_m": (np.abs(cells - 7) * 400)[None, :].astype("uint16"),
            "land_distance_m": (cells * 400)[None, :].astype("uint16"), "x0": 0.0, "y0": 400.0, "cell_m": 400.0}


def along_the_row(first_cell, last_cell):
    return [(first_cell * 400 + 200.0, 200.0), (last_cell * 400 + 200.0, 200.0)]


def test_map_classes_are_the_ones_of_the_detection_map():
    assert (rs.COAST_MARGIN, rs.NO_DATA) == (COAST_MARGIN, NO_DATA)


def test_north_pole_and_a_known_point_project_to_the_map():
    xy = rs.to_xy([(90.0, 0.0), (70.0, -45.0)])
    assert np.allclose(xy[0], 0.0, atol=1e-3)
    assert abs(xy[1, 0]) < 1e-3 and xy[1, 1] == pytest.approx(-2_194_000, rel=0.01)   # 20 degrees from the pole
    with pytest.raises(ValueError):
        rs.to_xy([(95.0, 10.0)])


def test_route_is_sampled_at_even_steps_and_ends_on_the_last_point():
    xy, dist = rs.sample_route([(0, 0), (1000, 0), (1000, 500)], step_m=400)
    assert dist.tolist() == [0, 400, 800, 1200, 1500]
    assert xy.tolist() == [[0, 0], [400, 0], [800, 0], [1000, 200], [1000, 500]]


@pytest.mark.parametrize("points", [[(0, 0)], [(5, 5), (5, 5)]])
def test_route_without_length_is_refused(points):
    with pytest.raises(ValueError):
        rs.sample_route(points)


def test_every_point_gets_the_right_status():
    out = rs.check_route(bundle(), along_the_row(0, 9), ship_m=100, ice_buffer_m=0, coast_margin_m=0)
    names = [rs.STATUS_NAMES[s] for s in out["status"]]
    assert names == ["land", "unverified", "clear", "clear", "clear", "clear", "clear", "ice", "unverified", "unverified"]
    assert out["scene"].tolist() == [255, 0, 0, 0, 0, 0, 0, 0, 255, 255]


def test_ice_buffer_and_half_the_ship_widen_the_ice():
    b = bundle()
    assert rs.check_route(b, along_the_row(5, 6), 100, 0, 0)["status"].tolist() == [rs.CLEAR, rs.CLEAR]
    assert rs.check_route(b, along_the_row(5, 6), 100, 400, 0)["status"].tolist() == [rs.CLEAR, rs.ICE_HIT]      # 400 < 450
    assert rs.check_route(b, along_the_row(5, 6), 100, 800, 0)["status"].tolist() == [rs.ICE_HIT, rs.ICE_HIT]
    assert rs.check_route(b, along_the_row(5, 6), 500, 600, 0)["status"].tolist() == [rs.ICE_HIT, rs.ICE_HIT]    # 250 + 600 >= 800


def test_ice_next_to_an_unseen_point_still_counts_and_land_beats_ice():
    b = bundle()
    assert rs.check_route(b, along_the_row(8, 9), 100, 400, 0)["status"].tolist() == [rs.ICE_HIT, rs.UNVERIFIED]
    assert rs.check_route(b, along_the_row(1, 2), 100, 20_000, 1000)["status"].tolist() == [rs.LAND_HIT, rs.LAND_HIT]


def test_coast_strip_is_never_called_clear_and_outside_the_map_is_unverified():
    b = bundle()
    assert rs.check_route(b, along_the_row(1, 1) + [(700.0, 200.0)], 100, 0, 0)["status"].tolist()[0] == rs.UNVERIFIED
    out = rs.check_route(b, [(-900.0, 200.0), (-100.0, 200.0)], 100, 0, 0)
    assert set(out["status"].tolist()) == {rs.UNVERIFIED} and set(out["scene"].tolist()) == {rs.NO_SCENE}


@pytest.mark.parametrize("kwargs", [dict(ship_m=0), dict(ship_m=float("nan")), dict(ice_buffer_m=-1), dict(coast_margin_m=50_000)])
def test_values_out_of_range_are_refused(kwargs):
    values = dict(ship_m=100, ice_buffer_m=0, coast_margin_m=0) | kwargs
    with pytest.raises(ValueError):
        rs.check_route(bundle(), along_the_row(2, 4), **values)


def test_verdict_blocked_names_the_first_conflict_with_its_time():
    dist = np.arange(0, 5) * 18_520.0                                    # 18.52 km apart = 1 h at 10 knots
    status = np.array([rs.CLEAR, rs.UNVERIFIED, rs.ICE_HIT, rs.LAND_HIT, rs.CLEAR], "uint8")
    v = rs.verdict(dist, status, speed_kn=10)
    assert v["state"] == "blocked"
    assert v["first_conflict"] == {"km": pytest.approx(37.04), "hours": pytest.approx(2.0), "kind": "ice"}
    assert v["total_km"] == pytest.approx(74.08) and v["hours"] == pytest.approx(4.0)
    assert v["km"]["clear"] == pytest.approx(18.52) and v["km"]["unverified"] == pytest.approx(18.52)
    assert sum(v["km"].values()) == pytest.approx(v["total_km"])


def test_verdict_never_says_clear_when_part_of_the_route_is_unseen():
    dist = np.arange(0, 4) * 1000.0
    clear, unseen = np.zeros(4, "uint8"), np.full(4, rs.UNVERIFIED, "uint8")
    assert rs.verdict(dist, clear, 10)["state"] == "clear"
    assert rs.verdict(dist, np.array([0, 1, 0, 0], "uint8"), 10)["state"] == "clear_where_observed"
    assert rs.verdict(dist, unseen, 10)["state"] == "no_data"
    assert rs.verdict(dist, clear, 10)["first_conflict"] is None
    with pytest.raises(ValueError):
        rs.verdict(dist, clear, speed_kn=0)
