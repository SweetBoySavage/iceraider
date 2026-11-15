"""Start the drift forecast from OSI SAF drift INTERPOLATED to finer points, then forecast.

    python scripts/forecast_osisaf_interp.py      # needs train_drift_model.py; part A also the NISAR scenes

Starting drift: the OSI SAF 62.5 km map of 26 Oct 12:00 -> 28 Oct 12:00 UTC 2026 (ends at the forecast start,
so it holds no information about the forecast days). It is interpolated linearly between its nominal-quality
cells (nearest cell outside them), then the drift model runs as usual:
    V1 = a*V0 + b*W1 + c*W0 + d, ...   (W: ERA5 wind, the wind that actually blew)

Part A - NISAR points. V0 is interpolated to the 176 Chukchi pieces of ice where the 27 -> 28 Oct NISAR arrows
end (28 Oct 10:20). Forecast 28 -> 29 and 29 -> 30 Oct at the NISAR pass times, checked against NISAR-tracked
drift exactly like forecast_chukchi.py, which starts from the NISAR drift instead. Same wind, same points.

Part B - fine grid. V0 is interpolated to a 10 km grid over the Chukchi / Beaufort box (inside the area of
valid OSI SAF cells). Forecast in 48 h steps on the fixed grid, with ERA5 wind interpolated the same way from the
OSI SAF cell centres. Checked against the OSI SAF drift of each window interpolated to the same points; also
scored at the OSI SAF cell centres. Because the model is linear and wind and drift are interpolated linearly,
the fine-grid forecast equals the 62.5 km forecast interpolated afterwards: it shows more points, not more
information.

Writes figures/osisaf_interp_nisar_check.png, figures/osisaf_interp_grid_forecast.png and
models/osisaf_interp_result.json.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_drift_training import BOX, NOMINAL, fetch_region           # noqa: E402
from scripts.drift_model import east_north_to_map, vector_errors              # noqa: E402
from scripts.forecast_chukchi import (CHAIN, COLORS, METHODS, Wind, background, forecast, forecast_from,  # noqa: E402
                                      score, track_pair)
from scripts.nisar_loader import load_gcov                                     # noqa: E402
from scripts.run_chukchi import DATA, FIGS, start_time                        # noqa: E402
from scripts.train_drift_model import load_models                             # noqa: E402
from scripts.wind import fetch_hourly, window_mean                             # noqa: E402

START = date(2026, 10, 26)                    # OSI SAF window 26 Oct 12:00 -> 28 Oct 12:00
FINE_KM = 10.0
GRID_STEPS = 6
TO_MAP = Transformer.from_crs(4326, 3413, always_xy=True)
OUT_JSON = ROOT / "models" / "osisaf_interp_result.json"


class OsiField:
    """One OSI SAF 48 h window: velocity km/day (complex, map frame), interpolated anywhere."""

    def __init__(self, start: date):
        f = fetch_region(start)
        if f is None:
            raise FileNotFoundError(f"no OSI SAF file for {start}")
        hrs = 48 + np.nan_to_num(f["dt1"] - f["dt0"]) / 3600
        v = (f["dX"] + 1j * f["dY"]) / (hrs / 24)
        self.lat, self.lon, self.v_grid = f["lat"], f["lon"], v
        self.ok = f["flag"] == NOMINAL
        self.in_box = (f["lat"] >= BOX["lat"][0]) & (f["lat"] <= BOX["lat"][1]) & (f["lon"] >= BOX["lon"][0]) & (f["lon"] <= BOX["lon"][1])
        px, py = TO_MAP.transform(f["lon"], f["lat"])
        self.px, self.py = px, py
        pts = np.column_stack([px[self.ok], py[self.ok]])
        self.lin, self.near = LinearNDInterpolator(pts, v[self.ok]), NearestNDInterpolator(pts, v[self.ok])
        self.start = start

    def at(self, x, y, fill=True):
        v = self.lin(x, y)
        return np.where(np.isfinite(v), v, self.near(x, y)) if fill else v


def part_a(models):
    pairs, scenes = [], {}
    for a, b in zip(CHAIN, CHAIN[1:]):
        for n in (a, b):
            scenes.setdefault(n, load_gcov(DATA / f"{n}.h5", pols=["hh"]))
        pairs.append(track_pair(scenes[a], scenes[b], a, b))
        scenes.pop(a)
    scenes.clear()
    day0, day1, day2 = pairs
    disp = day0["v"] * day0["hours"] / 24 * 1e3
    x0, y0 = day0["x"] + disp.real, day0["y"] + disp.imag              # the ice at 28 Oct 10:20
    osi = OsiField(START)
    v0_osi = osi.at(x0, y0)
    prev = (datetime(START.year, START.month, START.day, 12), datetime(START.year, START.month, START.day, 12) + timedelta(days=2))
    wind = Wind(day0["x"], day0["y"], f"{START - timedelta(days=1)}", f"{day2['t1'] + timedelta(days=1):%Y-%m-%d}")
    times = [day1["t0"], day1["t1"], day2["t1"]]
    v0_compare = vector_errors(v0_osi, day0["v"])
    print(f"Part A: {len(x0)} NISAR points. Starting drift: OSI SAF (interpolated) {np.median(np.abs(v0_osi)):.1f} km/day "
          f"vs NISAR {np.median(np.abs(day0['v'])):.1f} km/day; difference {v0_compare['rmse']:.1f} km/day, "
          f"median direction difference {v0_compare['median_abs_angle_deg']:.0f}°")
    out = {"start_drift_osisaf_vs_nisar": v0_compare, "osisaf_start": {}, "nisar_start": {}}
    runs = {}
    for start_name, run in (("osisaf_start", lambda m: forecast_from(m, models, x0, y0, v0_osi, prev, wind, times)),
                            ("nisar_start", lambda m: forecast(m, models, day0, wind, times))):
        for m in METHODS:
            steps = run(m)
            runs[(start_name, m)] = steps
            out[start_name][m] = {"day 1": score(steps[0], day1), "day 2": score(steps[1], day2)}
    print(f"\n  displacement error vs NISAR (km)   {'day 1':>8s} {'day 2':>8s}")
    for start_name, label in (("osisaf_start", "start: OSI SAF interpolated"), ("nisar_start", "start: NISAR drift")):
        for m in METHODS:
            r = out[start_name][m]
            print(f"  {label:28s} {m:12s} {r['day 1']['rmse']:8.2f} {r['day 2']['rmse']:8.2f}   (n = {r['day 1']['n']}, {r['day 2']['n']})")
    figure_a(day1, day2, runs, out)
    return out


def part_b(models):
    full, wind_only = models
    windows = [START + timedelta(days=2 * k) for k in range(GRID_STEPS + 1)]
    fields = [OsiField(d) for d in windows]
    f0 = fields[0]
    cells = f0.in_box & f0.ok
    # fine grid inside the valid start cells of the box
    xs = np.arange(f0.px[cells].min(), f0.px[cells].max() + 1, FINE_KM * 1e3)
    ys = np.arange(f0.py[cells].min(), f0.py[cells].max() + 1, FINE_KM * 1e3)
    gx, gy = np.meshgrid(xs, ys)
    inside = np.isfinite(LinearNDInterpolator(np.column_stack([f0.px[cells], f0.py[cells]]), np.ones(cells.sum()))(gx, gy))
    px, py = gx[inside], gy[inside]
    v0 = f0.at(px, py)

    # ERA5 wind at all box cell centres, interpolated to the fine points per window
    bc = f0.in_box
    t, east, north = fetch_hourly(f0.lat[bc], f0.lon[bc], f"{windows[0]}", f"{windows[-1] + timedelta(days=2)}")
    wmap = east_north_to_map(east, north, f0.lon[bc][:, None])
    pts = np.column_stack([f0.px[bc], f0.py[bc]])
    winds_fine, winds_cell, steps = [], [], 0
    for k, d in enumerate(windows):
        try:
            w = window_mean(t, wmap, f"{d}T12", f"{d + timedelta(days=2)}T12", complete=True)
        except ValueError:
            break
        winds_fine.append(LinearNDInterpolator(pts, w)(px, py))
        winds_cell.append(w[cells[bc]])
        steps = k
    print(f"\nPart B: {inside.sum()} fine-grid points ({FINE_KM:.0f} km) from {cells.sum()} OSI SAF cells; "
          f"{steps} steps of 48 h from {windows[1]} 12:00")

    pred = full.forecast(v0, winds_fine)
    pred_cell = full.forecast(f0.v_grid[cells], winds_cell)
    res = []
    print(f"\n  {'lead':>6s} {'window':>18s} {'obs speed':>10s} {'model, fine':>12s} {'persistence':>12s} {'model, cells':>13s}   (vector error, km/day)")
    for k in range(1, steps + 1):
        obs_fine = fields[k].at(px, py, fill=False)
        obs_cell = np.where(fields[k].ok[cells], fields[k].v_grid[cells], np.nan)
        s_fine = vector_errors(pred[k - 1], obs_fine)
        s_pers = vector_errors(v0, obs_fine)
        s_cell = vector_errors(pred_cell[k - 1], obs_cell)
        res.append({"window": f"{windows[k]}T12 -> {windows[k] + timedelta(days=2)}T12", "lead_days": 2 * k,
                    "fine_model": s_fine, "fine_persistence": s_pers, "cells_model": s_cell})
        print(f"  {'+' + str(2 * k) + ' d':>6s} {windows[k]:%d %b} -> {windows[k] + timedelta(days=2):%d %b} "
              f"{s_fine['mean_obs_speed']:10.1f} {s_fine['rmse']:12.2f} {s_pers['rmse']:12.2f} {s_cell['rmse']:13.2f}")
    figure_b(gx, gy, inside, pred, fields, cells, windows, steps, res, v0)
    return {"fine_km": FINE_KM, "points": int(inside.sum()), "steps": res}


def figure_a(day1, day2, runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    q = dict(angles="xy", scale_units="xy", scale=1, width=0.0022)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.8), layout="constrained")
    for ax, k, truth, label in ((axes[0], 0, day1, "Day 1"), (axes[1], 1, day2, "Day 2")):
        background(ax, truth["start"], truth["grid"])
        for steps, color in ((runs[("nisar_start", "full model")], "#ffd166"), (runs[("osisaf_start", "full model")], COLORS["full model"])):
            s = steps[k]
            d = s["v"] * s["hours"] / 24
            ax.quiver(s["x"] / 1e3, s["y"] / 1e3, d.real, d.imag, color=color, **q)
        d = truth["v"] * truth["hours"] / 24
        ax.quiver(truth["x"] / 1e3, truth["y"] / 1e3, d.real, d.imag, color=COLORS["NISAR"], **q)
        o, n = out["osisaf_start"]["full model"][f"day {k + 1}"], out["nisar_start"]["full model"][f"day {k + 1}"]
        ax.set_title(f"{label}: {truth['t0']:%d %b %H:%M} -> {truth['t1']:%d %b %H:%M} UTC ({k + 1}-day forecast)\n"
                     f"error vs NISAR ({o['n']} arrows): start from OSI SAF {o['rmse']:.1f} km, start from NISAR {n['rmse']:.1f} km",
                     fontsize=11)
        ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
        allx = np.concatenate([runs[("osisaf_start", "full model")][k]["x"], truth["x"]]) / 1e3
        ally = np.concatenate([runs[("osisaf_start", "full model")][k]["y"], truth["y"]]) / 1e3
        ax.set_xlim(allx.min() - 25, allx.max() + 25); ax.set_ylim(ally.min() - 25, ally.max() + 25)
    axes[0].legend(handles=[Line2D([], [], color=COLORS["NISAR"], lw=3, label="NISAR observed"),
                            Line2D([], [], color=COLORS["full model"], lw=3, label="forecast, start = OSI SAF interpolated"),
                            Line2D([], [], color="#ffd166", lw=3, label="forecast, start = NISAR drift")],
                   loc="lower left", framealpha=0.9)
    fig.suptitle("Forecast at the Chukchi NISAR points, started from OSI SAF drift (26 -> 28 Oct, 62.5 km) interpolated "
                 "to each point, vs started from the NISAR drift. Arrows = displacement at map scale.", fontsize=11)
    plt.savefig(FIGS / "osisaf_interp_nisar_check.png", dpi=90); plt.close()


def figure_b(gx, gy, inside, pred, fields, cells, windows, steps, res, v0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stretch, every = 2.0, 3                       # arrows 2x their 48 h displacement; one arrow per 30 km
    q = dict(angles="xy", scale_units="xy", scale=1 / stretch, width=0.004)
    n = steps + 1
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(6.3 * int(np.ceil(n / 2)), 12.4), squeeze=False, layout="constrained")
    ext = [gx.min() / 1e3, gx.max() / 1e3, gy.min() / 1e3, gy.max() / 1e3]
    thin = np.zeros(gx.shape, bool)
    thin[::every, ::every] = True
    pick = thin[inside]
    vmax = np.nanpercentile(np.abs(np.concatenate([v0] + pred)), 98)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        v = v0 if i == 0 else pred[i - 1]
        speed = np.full(gx.shape, np.nan)
        speed[inside] = np.abs(v)
        im = ax.imshow(speed, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        pxk, pyk = gx[inside][pick] / 1e3, gy[inside][pick] / 1e3
        d = v[pick] * 2
        ax.quiver(pxk, pyk, d.real, d.imag, color=COLORS["full model"] if i else "#444444", **q)
        f = fields[i]
        ok = f.ok & cells
        do = f.v_grid[ok] * 2
        ax.quiver(f.px[ok] / 1e3, f.py[ok] / 1e3, do.real, do.imag, color=COLORS["OSI SAF"], **{**q, "width": 0.003})
        win = f"{windows[i]:%d %b} 12:00 -> {windows[i] + timedelta(days=2):%d %b} 12:00"
        if i == 0:
            ax.set_title(f"START: OSI SAF {win}, interpolated to {FINE_KM:.0f} km\n(grey = interpolated, blue = OSI SAF cells)", fontsize=10.5)
        else:
            s = res[i - 1]
            ax.set_title(f"+{2 * i} days: {win}\nerror model {s['fine_model']['rmse']:.1f}, persistence "
                         f"{s['fine_persistence']['rmse']:.1f} km/day", fontsize=10.5)
        ax.set_aspect("equal"); ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
    fig.colorbar(im, ax=axes, shrink=0.5, label="forecast speed (km/day), 10 km grid")
    axes[0][0].legend(handles=[Line2D([], [], color=COLORS["OSI SAF"], lw=3, label="OSI SAF observed (62.5 km)"),
                               Line2D([], [], color=COLORS["full model"], lw=3, label="model forecast (every 3rd 10 km point)")],
                      loc="lower left", framealpha=0.9, fontsize=9)
    fig.suptitle(f"Drift forecast on a {FINE_KM:.0f} km grid, started from OSI SAF drift interpolated from 62.5 km; "
                 f"Chukchi / western Beaufort Sea. Arrows = 48 h displacement drawn {stretch:.0f}x; wind: ERA5", fontsize=11)
    plt.savefig(FIGS / "osisaf_interp_grid_forecast.png", dpi=80); plt.close()


def main():
    models = load_models()
    result = {"start_window": f"{START}T12 -> {START + timedelta(days=2)}T12"}
    result["part_b_fine_grid"] = part_b(models)
    result["part_a_nisar_points"] = part_a(models)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}, figures/osisaf_interp_nisar_check.png, figures/osisaf_interp_grid_forecast.png")


if __name__ == "__main__":
    main()
