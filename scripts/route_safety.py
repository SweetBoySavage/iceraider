"""Can a ship follow a route through the stitched ice map? All the rules of the interface are here.

The ship is a circle around its position with a diameter of the ship length. A point of the route is
- LAND_HIT    when land is closer than half the ship + the coast margin,
- ICE_HIT     when observed sea ice is closer than half the ship + the ice buffer,
- UNVERIFIED  when no radar image covers it (or it lies in the strip next to the coast, where the model is not valid),
- CLEAR       otherwise: observed open water, far enough from ice and land.

The ice does not move yet: every image is used as it was taken. Distances are measured on the map
(polar stereographic, true at 70 N), which is up to about 1 % off at 74 N.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

COAST_MARGIN, NO_DATA = 3, 255      # map classes, same numbers as scripts/evaluate_detection.py (kept apart so the
                                    # app starts with numpy and pyproj only; a test checks they stay equal)
CLEAR, UNVERIFIED, ICE_HIT, LAND_HIT = 0, 1, 2, 3
STATUS_NAMES = {CLEAR: "clear", UNVERIFIED: "unverified", ICE_HIT: "ice", LAND_HIT: "land"}
KNOT_KMH = 1.852
STEP_M = 400.0
LIMITS = {"ship_m": (1.0, 500.0), "speed_kn": (0.5, 40.0), "ice_buffer_m": (0.0, 20_000.0), "coast_margin_m": (0.0, 20_000.0)}
NO_SCENE = 255


def to_xy(latlon, epsg: int = 3413) -> np.ndarray:
    """(n, 2) array of map x, y in metres from (lat, lon) pairs in degrees."""
    latlon = np.asarray(latlon, float).reshape(-1, 2)
    if not (np.isfinite(latlon).all() and (np.abs(latlon[:, 0]) <= 90).all() and (np.abs(latlon[:, 1]) <= 360).all()):
        raise ValueError("route points must be (latitude, longitude) in degrees")
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(latlon[:, 1], latlon[:, 0])
    return np.column_stack([x, y])


def sample_route(points_xy, step_m: float = STEP_M) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step_m` along the route, plus the end point. Returns (xy (n, 2), distance from the start in m)."""
    points = np.asarray(points_xy, float).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("a route needs at least two points")
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    if along[-1] <= 0:
        raise ValueError("the route has no length")
    dist = np.append(np.arange(0.0, along[-1], step_m), along[-1])
    return np.column_stack([np.interp(dist, along, points[:, 0]), np.interp(dist, along, points[:, 1])]), dist


def check_inputs(**values: float) -> None:
    for key, value in values.items():
        low, high = LIMITS[key]
        if not (np.isfinite(value) and low <= value <= high):
            raise ValueError(f"{key} must be between {low:g} and {high:g}, got {value!r}")


def check_route(bundle: dict, points_xy, ship_m: float, ice_buffer_m: float, coast_margin_m: float,
                step_m: float = STEP_M) -> dict:
    """Status of every sampled point of the route. `bundle` is the map made by scripts/build_app_data.py."""
    check_inputs(ship_m=ship_m, ice_buffer_m=ice_buffer_m, coast_margin_m=coast_margin_m)
    xy, dist = sample_route(points_xy, step_m)
    classes = bundle["classes"]
    col = np.floor((xy[:, 0] - bundle["x0"]) / bundle["cell_m"]).astype(int)
    row = np.floor((bundle["y0"] - xy[:, 1]) / bundle["cell_m"]).astype(int)
    inside = (row >= 0) & (row < classes.shape[0]) & (col >= 0) & (col < classes.shape[1])
    r, c = row[inside], col[inside]

    status = np.full(len(xy), UNVERIFIED, "uint8")            # outside the map nothing is known
    scene = np.full(len(xy), NO_SCENE, "uint8")
    cls = classes[r, c]
    here = np.where((cls == NO_DATA) | (cls == COAST_MARGIN), UNVERIFIED, CLEAR)
    here[bundle["ice_distance_m"][r, c] <= ship_m / 2 + ice_buffer_m] = ICE_HIT
    here[bundle["land_distance_m"][r, c] <= ship_m / 2 + coast_margin_m] = LAND_HIT
    status[inside] = here
    scene[inside] = bundle["scene_index"][r, c]
    return {"xy": xy, "dist_m": dist, "status": status, "scene": scene}


def verdict(dist_m: np.ndarray, status: np.ndarray, speed_kn: float) -> dict:
    """Summary of a checked route: state, kilometres per status, and the first place where the ship is stopped."""
    check_inputs(speed_kn=speed_kn)
    speed_kmh = speed_kn * KNOT_KMH
    piece = np.diff(dist_m, append=dist_m[-1]) / 1e3          # km represented by every sample
    km = {name: float(piece[status == code].sum()) for code, name in STATUS_NAMES.items()}
    total = float(dist_m[-1] / 1e3)
    blocked = np.flatnonzero((status == ICE_HIT) | (status == LAND_HIT))
    first = None
    if blocked.size:
        i = int(blocked[0])
        first = {"km": float(dist_m[i] / 1e3), "hours": float(dist_m[i] / 1e3 / speed_kmh), "kind": STATUS_NAMES[int(status[i])]}
    if first:
        state = "blocked"
    elif km["clear"] <= 0:
        state = "no_data"
    elif km["unverified"] > 0:
        state = "clear_where_observed"
    else:
        state = "clear"
    return {"state": state, "total_km": total, "hours": total / speed_kmh, "km": km, "first_conflict": first}
