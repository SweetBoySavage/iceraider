"""Can a ship follow a route through the stitched ice map? All the rules of the interface are here.

The ship is a circle around its position with a diameter of the ship length. A point of the route is
- LAND_HIT    when land is closer than half the ship + the coast margin,
- ICE_HIT     when observed sea ice is closer than half the ship + the ice buffer,
- UNVERIFIED  when no radar image covers it (or it lies in the strip next to the coast, where the model is not valid),
- CLEAR       otherwise: observed open water, far enough from ice and land.

Static ice (no drift data, or moving switched off): every image is used as it was taken.
Moving ice (the bundle has a 'drift' forecast, made by scripts/build_app_drift.py): each sampled point is checked at the
moment the ship passes it, against the ice where the drift moved it by then:
- the ice of image s (taken at t_s) is at p + D(p, t_s -> t) at time t, with D the drift summed over the 48 h windows
  between t_s and t, read at the ice's own position p (the page draws the ice with the same rule);
- for a route point x at time t, the origin p = x - D(p) is found per image, newest image first; the first image whose
  own cell is hit gives the class and the distance to ice ("newest image wins", as in the stitched map);
- land, the strip next to the coast and the distance to land do not move;
- after the end of the forecast nothing is known about the ice: UNVERIFIED.
Distances are measured on the map (polar stereographic, true at 70 N), which is up to about 1 % off at 74 N.
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
LIMITS = {"ship_m": (1.0, 500.0), "speed_kn": (0.5, 40.0), "ice_buffer_m": (0.0, 20_000.0), "coast_margin_m": (0.0, 20_000.0),
          "delay_h": (0.0, 288.0)}
NO_SCENE = 255
ORIGIN_ITERATIONS = 4


def drift_velocity(drift: dict, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Drift (km/day along map x, y) per 48 h window at map points: bilinear between the 10 km nodes, edge nodes
    outside. Returns two (windows, n) arrays."""
    fx = np.clip((np.asarray(x, float) - drift["x0"]) / drift["node_m"], 0, drift["u"].shape[2] - 1.000001)
    fy = np.clip((drift["y0"] - np.asarray(y, float)) / drift["node_m"], 0, drift["u"].shape[1] - 1.000001)
    i, j = fx.astype(int), fy.astype(int)
    a, b = fx - i, fy - j
    out = []
    for f in (drift["u"], drift["v"]):
        out.append(f[:, j, i] * (1 - a) * (1 - b) + f[:, j, i + 1] * a * (1 - b) + f[:, j + 1, i] * (1 - a) * b + f[:, j + 1, i + 1] * a * b)
    return out[0], out[1]


def displacement(drift: dict, x, y, t_from, t_to) -> tuple[np.ndarray, np.ndarray]:
    """How far (metres along map x, y) ice at (x, y) at time t_from has moved at time t_to (seconds, UNIX).
    Negative when t_to is before t_from. Outside the drift windows the ice does not move."""
    u, v = drift_velocity(drift, x, y)
    t_from, t_to, _ = np.broadcast_arrays(np.asarray(t_from, float), np.asarray(t_to, float), np.asarray(x, float))
    lo, hi = np.minimum(t_from, t_to), np.maximum(t_from, t_to)
    edges = drift["edges_s"]
    days = np.stack([np.clip(np.minimum(hi, edges[w + 1]) - np.maximum(lo, edges[w]), 0, None) / 86_400
                     for w in range(len(edges) - 1)])                                  # (windows, n)
    sign = np.sign(t_to - t_from)
    return sign * (u * days).sum(0) * 1e3, sign * (v * days).sum(0) * 1e3


def origin_of(drift: dict, x, y, t_from, t_to, iterations: int = ORIGIN_ITERATIONS) -> tuple[np.ndarray, np.ndarray]:
    """Where the ice that is at (x, y) at time t_to was at time t_from: p with p + D(p, t_from -> t_to) = (x, y)."""
    px, py = np.asarray(x, float), np.asarray(y, float)
    for _ in range(iterations):
        dx, dy = displacement(drift, px, py, t_from, t_to)
        px, py = np.asarray(x, float) - dx, np.asarray(y, float) - dy
    return px, py


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


def cell_of(bundle: dict, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, inside the map) of map points."""
    col = np.floor((np.asarray(x) - bundle["x0"]) / bundle["cell_m"]).astype(int)
    row = np.floor((bundle["y0"] - np.asarray(y)) / bundle["cell_m"]).astype(int)
    shape = bundle["classes"].shape
    return row, col, (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])


def check_route(bundle: dict, points_xy, ship_m: float, ice_buffer_m: float, coast_margin_m: float,
                step_m: float = STEP_M, speed_kn: float | None = None, depart_s: float | None = None,
                moving: bool = True) -> dict:
    """Status of every sampled point of the route. `bundle` is the map made by scripts/build_app_data.py.

    With speed_kn and depart_s (UNIX seconds) and a bundle that has 'drift', the ice moves (see the module text);
    otherwise, or with moving=False, the ice is static."""
    check_inputs(ship_m=ship_m, ice_buffer_m=ice_buffer_m, coast_margin_m=coast_margin_m)
    xy, dist = sample_route(points_xy, step_m)
    if moving and bundle.get("drift") is not None and speed_kn is not None and depart_s is not None:
        check_inputs(speed_kn=speed_kn)
        times = depart_s + dist / (speed_kn * KNOT_KMH / 3.6)
        return _check_moving(bundle, xy, dist, times, ship_m, ice_buffer_m, coast_margin_m)
    classes = bundle["classes"]
    row, col, inside = cell_of(bundle, xy[:, 0], xy[:, 1])
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


def _check_moving(bundle, xy, dist, times, ship_m, ice_buffer_m, coast_margin_m) -> dict:
    drift, classes, scene_index = bundle["drift"], bundle["classes"], bundle["scene_index"]
    n = len(xy)
    found = np.zeros(n, bool)
    scene = np.full(n, NO_SCENE, "uint8")
    cls = np.full(n, NO_DATA, "uint8")
    ice_d = np.full(n, np.inf)
    unseen_ice_d = np.full(n, np.inf)               # ice next to a spot no image covers still counts, as in the static map
    for s in np.argsort(bundle["scene_time_s"])[::-1]:                           # newest image first
        px, py = origin_of(drift, xy[:, 0], xy[:, 1], bundle["scene_time_s"][s], times)
        row, col, inside = cell_of(bundle, px, py)
        own = np.zeros(n, bool)
        own[inside] = scene_index[row[inside], col[inside]] == s
        new = own & ~found
        scene[new], cls[new] = s, classes[row[new], col[new]]
        ice_d[new] = bundle["ice_distance_m"][row[new], col[new]]
        found |= new
        gap = np.zeros(n, bool)
        gap[inside] = scene_index[row[inside], col[inside]] == NO_SCENE
        unseen_ice_d[gap] = np.minimum(unseen_ice_d[gap], bundle["ice_distance_m"][row[gap], col[gap]])
    ice_d = np.where(found, ice_d, unseen_ice_d)

    status = np.full(n, UNVERIFIED, "uint8")
    status[found & (cls != NO_DATA) & (cls != COAST_MARGIN)] = CLEAR
    status[ice_d <= ship_m / 2 + ice_buffer_m] = ICE_HIT
    status[times > drift["horizon_s"]] = UNVERIFIED                              # no forecast: nothing known about the ice
    row, col, inside = cell_of(bundle, xy[:, 0], xy[:, 1])                       # land and the coast strip stay put
    static = np.full(n, NO_DATA, "uint8")
    static[inside] = classes[row[inside], col[inside]]
    status[(static == COAST_MARGIN) & (status == CLEAR)] = UNVERIFIED
    land_d = np.full(n, np.inf)
    land_d[inside] = bundle["land_distance_m"][row[inside], col[inside]]
    status[land_d <= ship_m / 2 + coast_margin_m] = LAND_HIT
    return {"xy": xy, "dist_m": dist, "status": status, "scene": scene, "time_s": times}


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
