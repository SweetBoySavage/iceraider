"""Grid forecast: start from one OSI SAF drift map and forecast every grid cell, 48 h at a time; check each step.

    python scripts/forecast_osisaf_grid.py                    # start window 27 Oct 12:00 -> 29 Oct 12:00, 2026
    python scripts/forecast_osisaf_grid.py --start 2026-10-15

Same product as the OSI SAF quicklooks (https://osisaf-hl.met.no/quicklooks-1prod?prod1=LR-Drift):
low-resolution sea-ice drift, 62.5 km grid, 48 h windows from 12:00 UTC. Area: the Chukchi / western
Beaufort box the drift model was trained on (70-80 N, 150-180 W). Needs build_drift_training.py (cached
OSI SAF maps) and train_drift_model.py (models/drift_model.json).

For every grid cell with nominal-quality drift in the start window (V0):
    V1 = a*V0 + b*W1 + c*W0 + d,   V2 = a*V1 + b*W2 + c*W1 + d,   ...   (each step uses the previous PREDICTION)
W = ERA5 10 m wind at the cell, averaged over the window (the wind that actually blew: a perfect wind
forecast). The grid stays fixed: each step is the drift forecast for that cell and window. It is compared
with the OSI SAF drift observed in the same cell and window (cells with nominal quality only), so forecast
and check have the same resolution. Baselines: persistence (V0 every step), wind only (b*W + d).
The fitted model never saw 2026.

Writes figures/osisaf_drift_forecast.png and models/osisaf_forecast_result.json.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_drift_training import BOX, NOMINAL, REGION_DIR      # noqa: E402
from scripts.drift_model import east_north_to_map, vector_errors      # noqa: E402
from scripts.forecast_chukchi import COLORS                            # noqa: E402
from scripts.train_drift_model import load_models                      # noqa: E402
from scripts.wind import fetch_hourly, window_mean                     # noqa: E402

FIGS = ROOT / "figures"
OUT_JSON = ROOT / "models" / "osisaf_forecast_result.json"
TO_MAP = Transformer.from_crs(4326, 3413, always_xy=True)
MAX_STEPS = 6
METHODS = ("full model", "wind only", "persistence")


def load_window(start: date):
    """One cached OSI SAF window (Chukchi box): velocity km/day (complex, map frame), nominal mask, lat, lon."""
    path = REGION_DIR / f"{start:%Y%m%d}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    if not z.files:
        return None
    hrs = 48 + np.nan_to_num(z["dt1"] - z["dt0"]) / 3600
    return dict(v=(z["dX"] + 1j * z["dY"]) / (hrs / 24), ok=z["flag"] == NOMINAL, lat=z["lat"], lon=z["lon"])


def main(start: date):
    full, wind_only = load_models()
    windows = [start + timedelta(days=2 * k) for k in range(MAX_STEPS + 1)]
    fields = [load_window(d) for d in windows]
    lat, lon = fields[0]["lat"], fields[0]["lon"]
    in_box = (lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1]) & (lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1])
    cells = in_box & fields[0]["ok"]                                    # where the start drift is known
    times, east, north = fetch_hourly(lat[cells], lon[cells], f"{windows[0]:%Y-%m-%d}", f"{windows[-1] + timedelta(days=2):%Y-%m-%d}")
    wmap = east_north_to_map(east, north, lon[cells][:, None])
    winds, steps = [], 0
    for k, d in enumerate(windows):
        try:
            w = window_mean(times, wmap, f"{d}T12", f"{d + timedelta(days=2)}T12", complete=True)
        except ValueError:
            break
        if not np.isfinite(w).all() or fields[k] is None:
            break
        winds.append(w)
        steps = k
    print(f"start window {windows[0]} 12:00 -> {windows[1]} 12:00 UTC: {cells.sum()} grid cells; "
          f"{steps} forecast steps of 48 h (ERA5 wind and OSI SAF maps available until {windows[steps] + timedelta(days=2)} 12:00)")

    v0 = fields[0]["v"][cells]
    pred = {"full model": full.forecast(v0, winds), "wind only": [], "persistence": [v0] * steps}
    for k in range(1, steps + 1):
        pred["wind only"].append(wind_only.predict(v0, winds[k], winds[k - 1]))
    result = {"start_window": f"{windows[0]}T12 -> {windows[1]}T12", "cells": int(cells.sum()), "steps": []}
    print(f"\n{'step':>4s} {'window':>25s} {'cells':>6s} {'obs speed':>10s} " + " ".join(f"{m:>12s}" for m in METHODS)
          + "    (vector error, km/day)")
    for k in range(1, steps + 1):
        obs = np.where(fields[k]["ok"][cells], fields[k]["v"][cells], np.nan)
        s = {m: vector_errors(pred[m][k - 1], obs) for m in METHODS}
        result["steps"].append({"window": f"{windows[k]}T12 -> {windows[k] + timedelta(days=2)}T12",
                                "lead_days": 2 * k, "wind_m_s": round(float(np.abs(winds[k]).mean()), 1),
                                **{m: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in s[m].items()} for m in METHODS}})
        print(f"{k:4d} {windows[k]:%d %b} -> {windows[k] + timedelta(days=2):%d %b} (+{2 * k:2d} d) {s['full model']['n']:6d} "
              f"{s['full model']['mean_obs_speed']:10.1f} " + " ".join(f"{s[m]['rmse']:12.2f}" for m in METHODS))
    OUT_JSON.write_text(json.dumps(result, indent=2))
    figure(lat, lon, cells, fields, pred, windows, steps, result)
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} and figures/osisaf_drift_forecast.png")


def figure(lat, lon, cells, fields, pred, windows, steps, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x, y = TO_MAP.transform(lon, lat)
    x, y = x / 1e3, y / 1e3
    stretch = 2.0                                                    # arrows drawn 2x their 48 h displacement
    q = dict(angles="xy", scale_units="xy", scale=1 / stretch, width=0.005)
    n = steps + 1
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 6.0 * rows), squeeze=False, layout="constrained")
    xs, ys = x[cells], y[cells]
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.set_facecolor("#f2f2f2")
        obs_ok = fields[i]["ok"][cells]
        d_obs = fields[i]["v"][cells] * 2                                # km over 48 h
        ax.quiver(xs[obs_ok], ys[obs_ok], d_obs[obs_ok].real, d_obs[obs_ok].imag, color=COLORS["OSI SAF"], **q)
        title = f"{windows[i]:%d %b} 12:00 -> {windows[i] + timedelta(days=2):%d %b} 12:00 UTC"
        if i == 0:
            ax.set_title(f"START (observed): {title}\n{cells.sum()} grid cells, drift used as V0", fontsize=11)
        else:
            d_fc = pred["full model"][i - 1] * 2
            ax.quiver(xs, ys, d_fc.real, d_fc.imag, color=COLORS["full model"], **q)
            s = result["steps"][i - 1]
            ax.set_title(f"+{2 * i} days: {title}\nerror model {s['full model']['rmse']:.1f}, persistence "
                         f"{s['persistence']['rmse']:.1f} km/day (wind {s['wind_m_s']:.0f} m/s)", fontsize=11)
        ax.set_xlim(xs.min() - 60, xs.max() + 60); ax.set_ylim(ys.min() - 60, ys.max() + 60)
        ax.set_aspect("equal"); ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
    axes[0][0].legend(handles=[Line2D([], [], color=COLORS["OSI SAF"], lw=3, label="OSI SAF observed"),
                               Line2D([], [], color=COLORS["full model"], lw=3, label="model forecast")],
                      loc="lower left", framealpha=0.9)
    fig.suptitle("Sea-ice drift forecast on the OSI SAF 62.5 km grid, Chukchi / western Beaufort Sea, from the observed "
                 f"drift of {windows[0]:%d %b %Y}. Arrows = displacement over 48 h, drawn {stretch:.0f}x longer; "
                 "wind: ERA5", fontsize=11)
    plt.savefig(FIGS / "osisaf_drift_forecast.png", dpi=80); plt.close()


if __name__ == "__main__":
    arg = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else "2026-10-27"
    main(datetime.strptime(arg, "%Y-%m-%d").date())
