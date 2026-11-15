"""Ice drift for the route interface: a 12-day forecast of how the ice of a scenario moves.

    python scripts/build_app_drift.py            # needs models/drift_model.json; downloads OSI SAF + ERA5 (cached)

For each scenario in DRIFT_SCENARIOS (October only: NISAR has no HH+VV here after about 1 Nov, so that is the
newest ice map) this reads app/data/<id>/meta.json (map grid, departure, image times) and runs
scripts/forecast_osisaf_interp.forecast_fields on a 10 km grid over the map:
- window 0: the observed OSI SAF drift map (48 h) that ends last at or before the departure (no look-ahead);
- windows 1..n: the drift model, 48 h steps, until at least departure + FORECAST_DAYS.
Wind = ERA5 (the wind that actually blew: a perfect wind forecast).

Writes app/data/<id>/drift.npz (read by the server for the safety check), drift.json + drift.bin (the same numbers
for the page: Float32, per window per node u then v, km/day along the map x and y axes).
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.forecast_osisaf_interp import forecast_fields              # noqa: E402
from scripts.run_app import APP_DATA                                     # noqa: E402

DRIFT_SCENARIOS = ("oct",)
FORECAST_DAYS = 12
NODE_M = 10_000.0
WINDOW_DAYS = 2
TO_LONLAT = Transformer.from_crs(3413, 4326, always_xy=True)


def choose_windows(departure: datetime, days: int = FORECAST_DAYS) -> tuple[date, int]:
    """(start of the observed OSI SAF window, number of 48 h forecast steps).

    The observed window is the last one (12:00 UTC to 12:00 UTC two days later) that ends at or before the
    departure; the forecast steps reach at least departure + days."""
    end = datetime(departure.year, departure.month, departure.day, 12, tzinfo=departure.tzinfo)
    if end > departure:
        end -= timedelta(days=1)
    start = end - timedelta(days=WINDOW_DAYS)
    steps = int(np.ceil((departure + timedelta(days=days) - end) / timedelta(days=WINDOW_DAYS)))
    return start.date(), steps


def node_grid(meta: dict, node_m: float = NODE_M) -> tuple[np.ndarray, np.ndarray]:
    """Node x (along the map x axis) and y (top to bottom) in metres, covering the whole map with one node to spare."""
    width, height = meta["nx"] * meta["cell_m"], meta["ny"] * meta["cell_m"]
    xs = meta["x0"] + np.arange(0, width + 2 * node_m, node_m) - node_m
    ys = meta["y0"] - np.arange(0, height + 2 * node_m, node_m) + node_m
    return xs, ys


def box_around(px, py, pad_deg: float = 1.0) -> dict:
    """A lat/lon box around map points; longitudes counted from the westmost point, so the date line is no problem."""
    lon, lat = TO_LONLAT.transform(px, py)
    lon = np.where(lon > 0, lon - 360, lon)
    return {"lat": (float(lat.min()) - pad_deg, min(float(lat.max()) + pad_deg, 89.0)),
            "lon": (float(lon.min()) - 2 * pad_deg, float(lon.max()) + 2 * pad_deg)}


def build(scenario_id: str, app_data: Path = APP_DATA) -> Path:
    folder = Path(app_data) / scenario_id
    meta = json.loads((folder / "meta.json").read_text())
    departure = datetime.fromisoformat(meta["route"]["departure_utc"].replace("Z", "+00:00"))
    start, steps = choose_windows(departure)
    xs, ys = node_grid(meta)
    gx, gy = np.meshgrid(xs, ys)
    out = forecast_fields(start, steps, gx.ravel(), gy.ravel(), box=box_around(gx, gy),
                          cache_dir=ROOT / "data" / "osisaf" / f"app_{scenario_id}")
    edges = [e.replace(tzinfo=timezone.utc) for e in out["edges"]]
    v = out["v"].reshape(len(edges) - 1, len(ys), len(xs))
    horizon = departure + timedelta(days=FORECAST_DAYS)
    info = {"node_m": NODE_M, "x0": float(xs[0]), "y0": float(ys[0]), "nx": len(xs), "ny": len(ys),
            "edges_utc": [f"{e:%Y-%m-%dT%H:%M:%SZ}" for e in edges],
            "windows": ["observed OSI SAF"] + ["forecast"] * steps,
            "horizon_utc": f"{horizon:%Y-%m-%dT%H:%M:%SZ}", "forecast_days": FORECAST_DAYS,
            "units": "km/day along the map x and y axes (EPSG:3413)",
            "source": "OSI SAF low-resolution drift (62.5 km, interpolated) + drift model with ERA5 wind"}
    np.savez_compressed(folder / "drift.npz", u=v.real.astype("float32"), v=v.imag.astype("float32"),
                        observed=out["observed"].reshape(len(ys), len(xs)),
                        edges_s=np.array([e.timestamp() for e in edges]), horizon_s=horizon.timestamp(),
                        x0=info["x0"], y0=info["y0"], node_m=NODE_M)
    (folder / "drift.json").write_text(json.dumps(info, indent=1))
    np.stack([v.real, v.imag], axis=1).astype("<f4").tofile(folder / "drift.bin")     # (window, u/v, row, col)
    print(f"{scenario_id}: {len(xs)} x {len(ys)} nodes, observed {edges[0]:%d %b %H:%M} -> {edges[1]:%d %b %H:%M}, "
          f"forecast to {edges[-1]:%d %b %H:%M} UTC; median speed per window "
          + ", ".join(f"{np.median(np.abs(w)):.1f}" for w in v) + " km/day")
    return folder


if __name__ == "__main__":
    for sid in DRIFT_SCENARIOS:
        build(sid)
    for f in sorted(APP_DATA.rglob("drift.*")):
        print(f"{f.stat().st_size / 1e6:6.2f} MB  {f.relative_to(ROOT)}")
