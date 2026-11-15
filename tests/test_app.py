import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_app_data as bad
from scripts import route_safety as rs
from scripts import run_app
from scripts.evaluate_detection import LAND, NO_DATA, OPEN_WATER, SEA_ICE

NAME = "NISAR_L2_PR_GCOV_034_045_D_051_0505_SVSH_A_20261027T074517_20261027T074554_P05036_N_F_J_001"
ROUTE = {"name": "test", "departure_utc": "2026-10-27T08:00:00Z", "points_lat_lon": [[72.0, -170.0], [72.0, -169.9]]}


def mosaic():
    """3 x 4 cells of 400 m around 72 N, 170 W: land on the left, then water, ice on the right, bottom row unseen."""
    x0, y0 = rs.to_xy([(72.0, -170.0)])[0]
    classes = np.array([[LAND, OPEN_WATER, OPEN_WATER, SEA_ICE]] * 2 + [[NO_DATA] * 4], "uint8")
    scene = np.where(classes == NO_DATA, 255, 0).astype("uint8")
    scene[:, 0] = 255
    ice_distance = np.array([[1200, 800, 400, 0]] * 2 + [[1265, 894, 565, 400]], "uint16")
    return {"classes": classes, "scene_index": scene, "ice_distance_m": ice_distance,
            "x": x0 - 400 + np.arange(4) * 400.0, "y": y0 + 400 - np.arange(3) * 400.0,
            "epsg": 3413, "cell_m": 400.0, "margin_m": 1000.0, "scenes": [NAME], "skipped": []}


def test_land_distance_is_in_metres_and_capped():
    classes = np.array([[LAND, OPEN_WATER, SEA_ICE, NO_DATA]], "uint8")
    assert bad.land_distance_m(classes, 400.0, cap_m=1000).tolist() == [[0, 400, 800, 1000]]
    assert bad.land_distance_m(classes[:, 1:], 400.0, cap_m=1000).tolist() == [[1000, 1000, 1000]]


def test_bundle_grid_starts_at_the_outer_corner_of_the_first_cell():
    m = mosaic()
    b = bad.make_bundle(m)
    assert b["x0"] == m["x"][0] - 200 and b["y0"] == m["y"][0] + 200
    assert b["land_distance_m"][0].tolist() == [0, 400, 800, 1200]


def test_pictures_carry_class_image_number_and_distances_in_steps_of_100_m():
    cells, distance = bad.display_images(bad.make_bundle(mosaic()))
    assert cells.shape == (3, 4, 3) and cells.dtype == np.uint8
    assert cells[0, :, 0].tolist() == [LAND, OPEN_WATER, OPEN_WATER, SEA_ICE] and cells[0, :, 1].tolist() == [255, 0, 0, 0]
    assert distance[0, :, 0].tolist() == [12, 8, 4, 0] and distance[0, :, 1].tolist() == [0, 4, 8, 12]
    assert distance[2, 0, 0] == 13                                       # 1265 m rounds up: the drawn margin never overstates


def test_written_files_load_again_and_meta_describes_them(tmp_path):
    bundle = bad.make_bundle(mosaic())
    bad.write(bundle, ROUTE, tmp_path)
    back = bad.load_bundle(tmp_path / "bundle.npz")
    assert all(np.array_equal(back[k], bundle[k]) for k in bad.ARRAYS)
    assert back["scenes"] == [NAME] and back["x0"] == bundle["x0"] and back["cell_m"] == 400.0
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert (meta["nx"], meta["ny"]) == (4, 3) and meta["route"] == ROUTE
    assert meta["scenes"] == [{"name": NAME, "time_utc": "2026-10-27T07:45:17Z", "col": 2.0, "row": 0.5}]
    assert meta["defaults"]["coast_margin_m"] == 1000.0 and meta["limits"]["ship_m"] == [1.0, 500.0]
    from PIL import Image
    assert Image.open(tmp_path / "cells.png").size == (4, 3) and Image.open(tmp_path / "distance.png").mode == "RGB"
    with pytest.raises(FileNotFoundError):
        bad.load_bundle(tmp_path / "nothing.npz")


def test_route_file_with_one_point_or_bad_numbers_is_refused(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(json.dumps(ROUTE | {"points_lat_lon": [[72.0, -170.0]]}))
    with pytest.raises(ValueError):
        bad.read_route(path)
    path.write_text(json.dumps(ROUTE | {"points_lat_lon": [[172.0, -170.0], [72.0, -170.0]]}))
    with pytest.raises(ValueError):
        bad.read_route(path)


def query_along_the_top_row():
    """A route from the first water cell (72 N, 170 W) two cells along the map x axis, onto the ice."""
    from pyproj import Transformer
    x, y = rs.to_xy([(72.0, -170.0)])[0]
    lon, lat = Transformer.from_crs(3413, 4326, always_xy=True).transform(x + 800, y)
    return f"ship_m=100&speed_kn=10&ice_buffer_m=0&coast_margin_m=0&route=72.0,-170.0;{lat:.7f},{lon:.7f}"


QUERY = query_along_the_top_row()


def test_query_is_parsed_and_broken_queries_are_refused():
    q = run_app.parse_query(QUERY)
    assert q["ship_m"] == 100 and q["speed_kn"] == 10 and q["route"][0] == (72.0, -170.0) and len(q["route"]) == 2
    for broken in ("ship_m=100", QUERY.replace("72.0,-170.0;", ""), QUERY.replace("ship_m=100", "ship_m=big"), QUERY + ";1,2,3"):
        with pytest.raises(ValueError):
            run_app.parse_query(broken)


def test_answer_gives_the_verdict_and_the_points_in_cell_positions():
    out = run_app.answer(bad.make_bundle(mosaic()), QUERY)
    assert out["verdict"]["state"] == "blocked" and out["verdict"]["first_conflict"]["kind"] == "ice"
    assert out["col"][0] == pytest.approx(1.5) and out["row"][0] == pytest.approx(1.5)     # the route starts in cell (1, 1)
    assert out["col"][-1] == pytest.approx(3.5, abs=0.01) and out["row"][-1] == pytest.approx(1.5, abs=0.01)
    assert out["status"] == [rs.CLEAR, rs.CLEAR, rs.ICE_HIT] and out["scene"] == [0, 0, 0]
    assert len(out["km"]) == len(out["col"]) == len(out["row"]) == len(out["status"]) == len(out["scene"])
    json.dumps(out)                                                       # everything must be plain JSON


def test_server_serves_the_page_and_answers_the_check(tmp_path, monkeypatch):
    bundle = bad.make_bundle(mosaic())
    monkeypatch.setattr(run_app, "APP_DATA", tmp_path)
    with pytest.raises(FileNotFoundError):
        run_app.make_server(0, bundle)                                    # no app data: refuse to start
    bad.write(bundle, ROUTE, tmp_path)
    server = run_app.make_server(0, bundle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"ICE RAIDER" in urllib.request.urlopen(f"{base}/").read()
        assert json.load(urllib.request.urlopen(f"{base}/api/check?{QUERY}"))["verdict"]["state"] == "blocked"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base}/api/check?{QUERY.replace('ship_m=100', 'ship_m=0')}")
        assert error.value.code == 400 and "ship_m" in json.load(error.value)["error"]
    finally:
        server.shutdown()
        server.server_close()
