"""Turn the stitched ice maps into the files the route interface loads. Run once after build_mosaic.py.

    python scripts/build_app_data.py

A scenario is a period with its own stitched map and route (config/scenarios.json). The app can switch between them.
Writes app/data/scenarios.json and, per scenario, app/data/<id>/ (small, committed to git, so the app starts
without the radar data):
- bundle.npz    what the safety check reads: classes, image per cell, distance to ice and to land in metres
- cells.png     for drawing: red = class, green = image number
- distance.png  for drawing the margins: red = distance to ice, green = distance to land, both in steps of 100 m
- meta.json     grid, image times, default values, the route of the scenario

Also writes app/data/world.json: a coarse outline of the land of the world (from the GSHHG coastline, LGPL), so the
page can show where on Earth the map lies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detection, land_mask, route_safety                         # noqa: E402
from scripts.build_mosaic import MAX_DISTANCE_M, load_mosaic                   # noqa: E402
from scripts.evaluate_detection import LAND                                    # noqa: E402
from scripts.run_app import APP_DATA, ARRAYS, load_bundle                      # noqa: E402,F401

SCENARIOS = ROOT / "config" / "scenarios.json"
PNG_STEP_M = 100                       # distance.png counts in steps of this size, 255 = 25.5 km or more
WORLD_STEP_DEG = 0.2                   # the world outline keeps one coast point about every 20 km
WORLD_MIN_COAST_DEG = 2.5              # islands with a shorter coast than this are left out of the world outline
DEFAULTS = {"ship_m": 200.0, "speed_kn": 10.0, "ice_buffer_m": 1000.0}


def land_distance_m(classes: np.ndarray, cell_m: float, cap_m: int = MAX_DISTANCE_M) -> np.ndarray:
    """Distance in metres from each cell to the nearest land cell (0 on land), capped at `cap_m`, as uint16."""
    land = classes == LAND
    if not land.any():
        return np.full(classes.shape, cap_m, "uint16")
    return np.minimum(distance_transform_edt(~land) * cell_m, cap_m).astype("uint16")


def thin_outline(lonlat: np.ndarray, step_deg: float) -> np.ndarray:
    """A coast polygon with about one point every `step_deg` (degrees along the coast, east-west scaled by latitude)."""
    lonlat = np.asarray(lonlat, float)
    d_lon = np.diff(lonlat[:, 0]) * np.cos(np.radians(lonlat[:-1, 1]))
    along = np.concatenate([[0.0], np.cumsum(np.hypot(d_lon, np.diff(lonlat[:, 1])))])
    keep = np.unique(np.searchsorted(along, np.arange(0.0, along[-1], step_deg)))
    return lonlat[keep]


def world_outline(polygons: list[np.ndarray], step_deg: float = WORLD_STEP_DEG, min_coast_deg: float = WORLD_MIN_COAST_DEG) -> list[list[float]]:
    """Coarse land polygons as flat lists [lon, lat, lon, lat, ...], two decimals. Small islands are dropped."""
    out = []
    for lonlat in polygons:
        thin = thin_outline(lonlat, step_deg)
        if len(thin) >= 4 and (len(thin) - 1) * step_deg >= min_coast_deg:
            out.append(np.round(thin, 2).ravel().tolist())
    return out


def write_world(app_data: Path = APP_DATA, coast_file: Path = land_mask.GSHHG_FILE) -> Path:
    if not Path(coast_file).exists():
        raise FileNotFoundError(f"{coast_file} is missing. Run: python -c 'from scripts.land_mask import download; download()'")
    world = {"source": "GSHHG 2.3.7 coastline (Wessel and Smith), LGPL, thinned", "polygons": world_outline(land_mask.read_gshhg(coast_file))}
    out = Path(app_data) / "world.json"
    out.write_text(json.dumps(world, separators=(",", ":")))
    return out


def make_bundle(mosaic: dict) -> dict:
    cell = float(mosaic["cell_m"])
    return {"classes": mosaic["classes"], "scene_index": mosaic["scene_index"], "ice_distance_m": mosaic["ice_distance_m"],
            "land_distance_m": land_distance_m(mosaic["classes"], cell),
            "x0": float(mosaic["x"][0] - cell / 2), "y0": float(mosaic["y"][0] + cell / 2), "cell_m": cell,
            "epsg": int(mosaic["epsg"]), "margin_m": float(mosaic["margin_m"]),
            "scenes": list(mosaic["scenes"]), "skipped": list(mosaic["skipped"])}


def display_images(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Two (rows, cols, 3) uint8 pictures: cells (class, image number, 0) and distance (to ice, to land, 0)."""
    def steps(metres):
        return np.minimum(np.ceil(metres / PNG_STEP_M), 255).astype("uint8")
    zero = np.zeros(bundle["classes"].shape, "uint8")
    cells = np.dstack([bundle["classes"], bundle["scene_index"], zero])
    distance = np.dstack([steps(bundle["ice_distance_m"]), steps(bundle["land_distance_m"]), zero])
    return cells, distance


def scene_list(bundle: dict) -> list[dict]:
    """Per used image: name, time, and the centre (col, row) of the cells it fills on the map."""
    out = []
    for i, name in enumerate(bundle["scenes"]):
        rows, cols = np.nonzero(bundle["scene_index"] == i)
        centre = {"col": float(cols.mean()), "row": float(rows.mean())} if rows.size else {"col": None, "row": None}
        out.append({"name": name, "time_utc": f"{detection.scene_time(name):%Y-%m-%dT%H:%M:%SZ}", **centre})
    return out


def read_scenarios(path: Path = SCENARIOS) -> list[dict]:
    """The scenarios of config/scenarios.json. Fails early on a broken file."""
    scenarios = json.loads(Path(path).read_text())
    if not scenarios:
        raise ValueError(f"{path}: no scenarios")
    ids = [s["id"] for s in scenarios]
    if len(set(ids)) != len(ids) or not all(i.isalnum() for i in ids):
        raise ValueError(f"{path}: scenario ids must be unique and made of letters and digits, got {ids}")
    for s in scenarios:
        if len(s["route"]["points_lat_lon"]) < 2:
            raise ValueError(f"{path}: the route of '{s['id']}' needs at least two points")
        route_safety.to_xy(s["route"]["points_lat_lon"])
        missing = {"label", "first_day", "last_day", "note"} - set(s) | {"name", "departure_utc"} - set(s["route"])
        if missing:
            raise ValueError(f"{path}: scenario '{s['id']}' misses {sorted(missing)}")
    return scenarios


def make_meta(bundle: dict, scenario: dict) -> dict:
    rows, cols = bundle["classes"].shape
    return {"id": scenario["id"], "label": scenario["label"], "note": scenario["note"],
            "epsg": bundle["epsg"], "x0": bundle["x0"], "y0": bundle["y0"], "cell_m": bundle["cell_m"], "nx": cols, "ny": rows,
            "png_step_m": PNG_STEP_M, "scenes": scene_list(bundle),
            "skipped": [{"name": n, "time_utc": f"{detection.scene_time(n):%Y-%m-%dT%H:%M:%SZ}"} for n in bundle["skipped"]],
            "defaults": DEFAULTS | {"coast_margin_m": bundle["margin_m"]},
            "limits": {k: list(v) for k, v in route_safety.LIMITS.items()}, "route": scenario["route"]}


def write(bundle: dict, scenario: dict, app_data: Path = APP_DATA) -> Path:
    """Write the files of one scenario into app_data/<id>/. Returns that folder."""
    from PIL import Image
    out_dir = Path(app_data) / scenario["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "bundle.npz", **bundle)
    for name, picture in zip(("cells.png", "distance.png"), display_images(bundle)):
        Image.fromarray(picture, "RGB").save(out_dir / name, optimize=True)
    (out_dir / "meta.json").write_text(json.dumps(make_meta(bundle, scenario), indent=1, ensure_ascii=False))
    return out_dir


def write_index(scenarios: list[dict], app_data: Path = APP_DATA) -> None:
    """The list the app and the server start from: the first scenario is the one shown first."""
    index = [{"id": s["id"], "label": s["label"]} for s in scenarios]
    (Path(app_data) / "scenarios.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    scenarios = read_scenarios()
    for scenario in scenarios:
        write(make_bundle(load_mosaic(scenario["first_day"], scenario["last_day"])), scenario)
    write_index(scenarios)
    write_world()
    for f in sorted(p for p in APP_DATA.rglob("*") if p.is_file()):
        print(f"{f.stat().st_size / 1e6:6.2f} MB  {f.relative_to(ROOT)}")
