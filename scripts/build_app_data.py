"""Turn the stitched ice map into the files the route interface loads. Run once after build_mosaic.py.

    python scripts/build_app_data.py

Writes into app/data/ (small, committed to git, so the app starts without the radar data):
- bundle.npz    what the safety check reads: classes, image per cell, distance to ice and to land in metres
- cells.png     for drawing: red = class, green = image number
- distance.png  for drawing the margins: red = distance to ice, green = distance to land, both in steps of 100 m
- meta.json     grid, image times, default values, the default route (config/route.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detection, route_safety                                    # noqa: E402
from scripts.build_mosaic import FIRST_DAY, LAST_DAY, MAX_DISTANCE_M, load_mosaic   # noqa: E402
from scripts.evaluate_detection import LAND                                    # noqa: E402
from scripts.run_app import APP_DATA, ARRAYS, load_bundle                      # noqa: E402,F401

ROUTE = ROOT / "config" / "route.json"
PNG_STEP_M = 100                       # distance.png counts in steps of this size, 255 = 25.5 km or more
DEFAULTS = {"ship_m": 200.0, "speed_kn": 10.0, "ice_buffer_m": 1000.0}


def land_distance_m(classes: np.ndarray, cell_m: float, cap_m: int = MAX_DISTANCE_M) -> np.ndarray:
    """Distance in metres from each cell to the nearest land cell (0 on land), capped at `cap_m`, as uint16."""
    land = classes == LAND
    if not land.any():
        return np.full(classes.shape, cap_m, "uint16")
    return np.minimum(distance_transform_edt(~land) * cell_m, cap_m).astype("uint16")


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


def read_route(path: Path = ROUTE) -> dict:
    route = json.loads(Path(path).read_text())
    route_safety.to_xy(route["points_lat_lon"])            # fails early on a broken file
    if len(route["points_lat_lon"]) < 2:
        raise ValueError(f"{path}: a route needs at least two points")
    return route


def make_meta(bundle: dict, route: dict) -> dict:
    rows, cols = bundle["classes"].shape
    return {"epsg": bundle["epsg"], "x0": bundle["x0"], "y0": bundle["y0"], "cell_m": bundle["cell_m"], "nx": cols, "ny": rows,
            "png_step_m": PNG_STEP_M, "scenes": scene_list(bundle),
            "skipped": [{"name": n, "time_utc": f"{detection.scene_time(n):%Y-%m-%dT%H:%M:%SZ}"} for n in bundle["skipped"]],
            "defaults": DEFAULTS | {"coast_margin_m": bundle["margin_m"]},
            "limits": {k: list(v) for k, v in route_safety.LIMITS.items()}, "route": route}


def write(bundle: dict, route: dict, out_dir: Path = APP_DATA) -> None:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "bundle.npz", **bundle)
    for name, picture in zip(("cells.png", "distance.png"), display_images(bundle)):
        Image.fromarray(picture, "RGB").save(out_dir / name, optimize=True)
    (out_dir / "meta.json").write_text(json.dumps(make_meta(bundle, route), indent=1))


if __name__ == "__main__":
    write(make_bundle(load_mosaic(FIRST_DAY, LAST_DAY)), read_route())
    for f in sorted(APP_DATA.iterdir()):
        print(f"{f.stat().st_size / 1e6:6.2f} MB  {f.relative_to(ROOT)}")
