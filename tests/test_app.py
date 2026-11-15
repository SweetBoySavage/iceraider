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
SCENARIO = {"id": "demo", "label": "Demo", "note": "A note", "first_day": "20261027", "last_day": "20261031", "route": ROUTE}


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


def test_world_outline_thins_the_coast_and_drops_small_islands():
    angle = np.linspace(0, 2 * np.pi, 2001)
    big = np.column_stack([10 * np.cos(angle), 10 * np.sin(angle)])              # circle of 10 degrees around 0 N, 0 E
    small = np.column_stack([100 + 0.1 * np.cos(angle), 0.1 * np.sin(angle)])
    thin = bad.thin_outline(big, 1.0)
    assert 55 <= len(thin) <= 66 and np.allclose(np.hypot(*thin.T), 10)         # about one point per degree, all on the coast
    world = bad.world_outline([big, small], step_deg=1.0, min_coast_deg=5.0)
    assert len(world) == 1 and len(world[0]) == 2 * len(thin) and world[0][:2] == [10.0, 0.0]


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
    folder = bad.write(bundle, SCENARIO, tmp_path)
    bad.write_index([SCENARIO], tmp_path)
    assert folder == tmp_path / "demo" and json.loads((tmp_path / "scenarios.json").read_text()) == [{"id": "demo", "label": "Demo"}]
    back = run_app.load_bundles(tmp_path)["demo"]
    assert all(np.array_equal(back[k], bundle[k]) for k in bad.ARRAYS)
    assert back["scenes"] == [NAME] and back["x0"] == bundle["x0"] and back["cell_m"] == 400.0
    meta = json.loads((folder / "meta.json").read_text())
    assert (meta["nx"], meta["ny"]) == (4, 3) and meta["route"] == ROUTE and (meta["id"], meta["note"]) == ("demo", "A note")
    assert meta["scenes"] == [{"name": NAME, "time_utc": "2026-10-27T07:45:17Z", "col": 2.0, "row": 0.5}]
    assert meta["defaults"]["coast_margin_m"] == 1000.0 and meta["limits"]["ship_m"] == [1.0, 500.0]
    from PIL import Image
    assert Image.open(folder / "cells.png").size == (4, 3) and Image.open(folder / "distance.png").mode == "RGB"


def test_app_refuses_to_start_without_its_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_app.load_bundles(tmp_path)                                    # no list of scenarios
    folder = bad.write(bad.make_bundle(mosaic()), SCENARIO, tmp_path)
    bad.write_index([SCENARIO], tmp_path)
    (folder / "cells.png").unlink()
    with pytest.raises(FileNotFoundError):
        run_app.load_bundles(tmp_path)


@pytest.mark.parametrize("change", [
    lambda s: s["route"].update(points_lat_lon=[[72.0, -170.0]]),                       # one point
    lambda s: s["route"].update(points_lat_lon=[[172.0, -170.0], [72.0, -170.0]]),      # no such latitude
    lambda s: s.update(id="de mo"),
    lambda s: s.pop("note"),
    lambda s: s["route"].pop("departure_utc"),
])
def test_broken_scenario_file_is_refused(tmp_path, change):
    path = tmp_path / "scenarios.json"
    scenario = json.loads(json.dumps(SCENARIO))
    path.write_text(json.dumps([scenario]))
    assert bad.read_scenarios(path)[0]["id"] == "demo"
    change(scenario)
    path.write_text(json.dumps([scenario]))
    with pytest.raises(ValueError):
        bad.read_scenarios(path)


def test_scenario_ids_must_be_unique_and_the_real_file_is_valid(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([SCENARIO, SCENARIO]))
    with pytest.raises(ValueError):
        bad.read_scenarios(path)
    assert [s["id"] for s in bad.read_scenarios()] == ["oct", "sep"]


def query_along_the_top_row():
    """A route from the first water cell (72 N, 170 W) two cells along the map x axis, onto the ice."""
    from pyproj import Transformer
    x, y = rs.to_xy([(72.0, -170.0)])[0]
    lon, lat = Transformer.from_crs(3413, 4326, always_xy=True).transform(x + 800, y)
    return f"scenario=demo&ship_m=100&speed_kn=10&ice_buffer_m=0&coast_margin_m=0&route=72.0,-170.0;{lat:.7f},{lon:.7f}"


QUERY = query_along_the_top_row()


def test_query_is_parsed_and_broken_queries_are_refused():
    q = run_app.parse_query(QUERY)
    assert q["scenario"] == "demo" and q["ship_m"] == 100 and q["speed_kn"] == 10 and q["route"][0] == (72.0, -170.0) and len(q["route"]) == 2
    for broken in ("ship_m=100", QUERY.replace("72.0,-170.0;", ""), QUERY.replace("ship_m=100", "ship_m=big"), QUERY + ";1,2,3",
                   QUERY.replace("scenario=demo&", "")):
        with pytest.raises(ValueError):
            run_app.parse_query(broken)


def test_answer_gives_the_verdict_and_the_points_in_cell_positions():
    bundles = {"demo": bad.make_bundle(mosaic())}
    with pytest.raises(ValueError):
        run_app.answer(bundles, QUERY.replace("scenario=demo", "scenario=other"))
    out = run_app.answer(bundles, QUERY)
    assert out["verdict"]["state"] == "blocked" and out["verdict"]["first_conflict"]["kind"] == "ice"
    assert out["col"][0] == pytest.approx(1.5) and out["row"][0] == pytest.approx(1.5)     # the route starts in cell (1, 1)
    assert out["col"][-1] == pytest.approx(3.5, abs=0.01) and out["row"][-1] == pytest.approx(1.5, abs=0.01)
    assert out["status"] == [rs.CLEAR, rs.CLEAR, rs.ICE_HIT] and out["scene"] == [0, 0, 0]
    assert len(out["km"]) == len(out["col"]) == len(out["row"]) == len(out["status"]) == len(out["scene"])
    json.dumps(out)                                                       # everything must be plain JSON


def test_server_serves_the_page_and_answers_the_check():
    server = run_app.make_server(0, {"demo": bad.make_bundle(mosaic())})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/")
        html = page.read()
        assert b"ICE RAIDER" in html and b"not for real navigation" in html
        assert page.headers["Cache-Control"] == "no-cache"
        again = urllib.request.Request(f"{base}/", headers={"If-Modified-Since": page.headers["Last-Modified"]})
        with pytest.raises(urllib.error.HTTPError) as unchanged:           # an unchanged file is not sent twice
            urllib.request.urlopen(again)
        assert unchanged.value.code == 304
        assert json.load(urllib.request.urlopen(f"{base}/api/check?{QUERY}"))["verdict"]["state"] == "blocked"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base}/api/check?{QUERY.replace('ship_m=100', 'ship_m=0')}")
        assert error.value.code == 400 and "ship_m" in json.load(error.value)["error"]
    finally:
        server.shutdown()
        server.server_close()


# ---------- moving ice ----------
def test_drift_windows_never_look_past_the_departure_and_cover_12_days():
    from datetime import date, datetime, timezone
    from scripts.build_app_drift import choose_windows
    assert choose_windows(datetime(2026, 10, 27, 8, tzinfo=timezone.utc)) == (date(2026, 10, 24), 7)   # ends 26 Oct 12:00
    assert choose_windows(datetime(2026, 10, 27, 12, tzinfo=timezone.utc)) == (date(2026, 10, 25), 6)  # ends exactly at departure


def small_drift(folder):
    """A uniform drift of 10 km/day along map x over the test map, one window of 30 days from 27 Oct."""
    start = run_app.utc_seconds("2026-10-27T00:00:00Z")
    np.savez(folder / "drift.npz", u=np.full((1, 2, 2), 10.0, "float32"), v=np.zeros((1, 2, 2), "float32"),
             observed=np.ones((2, 2), bool), edges_s=np.array([start, start + 30 * 86_400.0]), horizon_s=start + 12 * 86_400.0,
             x0=-3e6, y0=3e6, node_m=6e6)


def test_drift_is_loaded_when_present_and_the_check_takes_delay_and_moving(tmp_path):
    folder = bad.write(bad.make_bundle(mosaic()), SCENARIO, tmp_path)
    bad.write_index([SCENARIO], tmp_path)
    assert run_app.load_bundles(tmp_path)["demo"]["drift"] is None                  # no drift file: static, as before
    small_drift(folder)
    bundles = run_app.load_bundles(tmp_path)
    assert bundles["demo"]["drift"]["u"].shape == (1, 2, 2) and bundles["demo"]["scene_time_s"].shape == (1,)
    moving = run_app.answer(bundles, QUERY)
    static = run_app.answer(bundles, QUERY + "&moving=0")
    assert moving["moving"] and not static["moving"] and static["status"] == [rs.CLEAR, rs.CLEAR, rs.ICE_HIT]
    assert moving["forecast_end_h"] == pytest.approx(12 * 24 - 8)                     # departure 27 Oct 08:00
    later = run_app.answer(bundles, QUERY + "&delay_h=48")
    assert later["delay_h"] == 48 and later["status"] != static["status"]           # 2 days later the ice has moved 20 km away
    with pytest.raises(ValueError):
        run_app.answer(bundles, QUERY + "&delay_h=400")


def test_real_october_map_has_a_drift_forecast_and_both_checks_run():
    bundles = run_app.load_bundles()
    assert bundles["oct"]["drift"] is not None and bundles["sep"]["drift"] is None
    route = ";".join(f"{lat},{lon}" for lat, lon in json.loads((run_app.APP_DATA / "oct" / "meta.json").read_text())["route"]["points_lat_lon"])
    base = f"scenario=oct&ship_m=200&speed_kn=10&ice_buffer_m=1000&coast_margin_m=1000&route={route}"
    for extra in ("", "&moving=0", "&delay_h=120"):
        out = run_app.answer(bundles, base + extra)
        assert out["verdict"]["state"] in ("blocked", "clear", "clear_where_observed", "no_data")


def test_page_and_server_move_the_ice_the_same_way():
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    rng = np.random.default_rng(0)
    w, ny, nx = 3, 4, 5
    data = rng.normal(0, 8, (w, 2, ny, nx)).astype("float32")
    d = {"u": data[:, 0], "v": data[:, 1], "x0": -50_000.0, "y0": 40_000.0, "node_m": 10_000.0,
         "edges_s": np.array([0.0, 2, 4, 6]) * 86_400}
    pts = [(-37_000.0, 12_000.0, 0.3, 4.4), (5_000.0, -3_000.0, 5.5, 1.0), (-90_000.0, 90_000.0, -1.0, 7.0)]
    py = [rs.displacement(d, [x], [y], a * 86_400, b * 86_400) for x, y, a, b in pts]
    script = ("const D = require(%s); const d = %s; d.data = new Float32Array(d.data);"
              "console.log(JSON.stringify(%s.map(([x, y, a, b]) => D.displacement(d, x, y, a * 86400, b * 86400))));") % (
        json.dumps(str(run_app.APP / "drift.js")),
        json.dumps({"x0": d["x0"], "y0": d["y0"], "node_m": d["node_m"], "nx": nx, "ny": ny,
                    "edges_s": d["edges_s"].tolist(), "data": data.ravel().tolist()}), json.dumps(pts))
    js = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    for (dx, dy), (jx, jy) in zip(py, js):
        assert jx == pytest.approx(dx[0], abs=0.05) and jy == pytest.approx(dy[0], abs=0.05)
