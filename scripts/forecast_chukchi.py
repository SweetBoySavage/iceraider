"""Forecast the NISAR drift arrows 1, 2 and 7 days ahead, and check them.

    python scripts/forecast_chukchi.py      # after download_scenes.py, build_drift_training.py, train_drift_model.py

1. Track three NISAR pairs with the same method as run_chukchi.py (radar texture, similarity >= 0.5):
       day 0  27 Oct 11:01 -> 28 Oct 10:20   (the input: "previous movement", 176 arrows)
       day 1  28 Oct 10:20 -> 29 Oct 09:39   (truth for the 1-day forecast)
       day 2  29 Oct 09:39 -> 30 Oct 10:37   (truth for the 2-day forecast)
2. ERA5 hourly 10 m wind on a 0.5 deg x 1 deg grid around the ice, averaged over each step.
3. Forecast every day-0 arrow step by step (forecast()): the first step starts at the END of the day-0 arrow,
   each next step starts where the previous predicted step ended, and its "previous drift" is the previous
   prediction. Wind = ERA5 reanalysis, i.e. a perfect wind forecast.
       - 2 steps at the NISAR pass times  -> compared with the NISAR arrows that start within MATCH_KM
       - 7 steps of 24 h from 28 Oct 10:20 -> compared with trajectories through the OSI SAF drift fields
         of the same days (62.5 km, independent: 2026 is not used for fitting)
   Baselines: persistence (keep the day-0 drift) and wind only (b*W + d).

Writes figures/chukchi_drift_forecast.png, figures/chukchi_drift_forecast_7day.png,
models/drift_forecast_result.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_scenes import FORECAST_SCENES, SCENES            # noqa: E402
from scripts.drift_model import east_north_to_map, vector_errors       # noqa: E402
from scripts.ice_tracking import track                                  # noqa: E402
from scripts.nisar_loader import load_gcov                              # noqa: E402
from scripts.run_chukchi import DATA, EPSG, FIGS, SPACING, TRACK, common_grid, on_grid, start_time, texture, to_db  # noqa: E402
from scripts.train_drift_model import load_models                       # noqa: E402
from scripts.wind import fetch_hourly, window_mean                      # noqa: E402

CHAIN = [s for s in SCENES if s.startswith("NISAR")][:2] + FORECAST_SCENES   # 27, 28, 29, 30 Oct
MATCH_KM = 4.0           # a forecast arrow and a NISAR arrow describe the same ice if their starts are this close
DAYS = 7
OSI_REGION = ROOT / "data" / "osisaf" / "region"                    # cached by build_drift_training.py
OUT_JSON = ROOT / "models" / "drift_forecast_result.json"
TO_LONLAT = Transformer.from_crs(EPSG, 4326, always_xy=True)
TO_MAP = Transformer.from_crs(4326, EPSG, always_xy=True)
METHODS = ("full model", "wind only", "persistence")
COLORS = {"NISAR": "#00b4d8", "full model": "#ff7f0e", "OSI SAF": "#2166ac", "persistence": "#9e9e9e"}


def hours(a: str, b: str) -> float:
    return (start_time(b) - start_time(a)).total_seconds() / 3600


def track_pair(sa, sb, a: str, b: str, pol: str = "hh", track_kw: dict = TRACK) -> dict:
    """NISAR drift from scene a to b: arrow starts (map m) and velocity (km/day, complex map vector)."""
    x, y = common_grid([sa, sb])
    t = track(texture(on_grid(sa, pol, x, y)), texture(on_grid(sb, pol, x, y)), **track_kw)
    disp_km = (t["d_col"] - 1j * t["d_row"]) * SPACING / 1e3            # rows grow southwards
    h = hours(a, b)
    return dict(x=x[t["col"].astype(int)], y=y[t["row"].astype(int)], v=disp_km / (h / 24), hours=h,
                start=a, end=b, t0=start_time(a), t1=start_time(b), grid=(x, y))


def moved(x, y, v, h):
    """Position after moving with velocity v (km/day, complex) for h hours. Positions in map metres."""
    d = v * h / 24 * 1e3
    return x + d.real, y + d.imag


class Wind:
    """ERA5 wind (map-frame complex, m/s) on a lat/lon grid around the ice; nearest grid point is used."""

    def __init__(self, xs, ys, start_date, end_date):
        lon, lat = TO_LONLAT.transform(xs, ys)
        lats = np.arange(np.floor(np.min(lat)) - 1.0, min(np.ceil(np.max(lat)) + 2.5, 85.0) + 0.01, 0.5)
        lons = np.arange(np.floor(np.min(lon)) - 4.0, np.ceil(np.max(lon)) + 10.0 + 0.01, 1.0)
        glat, glon = [g.ravel() for g in np.meshgrid(lats, lons, indexing="ij")]
        self.times, east, north = fetch_hourly(glat, glon, start_date, end_date)
        self.map = east_north_to_map(east, north, glon[:, None])
        px, py = TO_MAP.transform(glon, glat)
        self.tree = cKDTree(np.column_stack([px, py]))

    def mean(self, x, y, t0: datetime, t1: datetime):
        """Mean wind between t0 and t1 at map positions (x, y)."""
        _, idx = self.tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        w = window_mean(self.times, self.map, f"{t0:%Y-%m-%dT%H}", f"{t1:%Y-%m-%dT%H}", complete=True)[idx]
        if not np.isfinite(w).all():
            raise ValueError(f"ERA5 wind missing between {t0} and {t1}")
        return w


def forecast(model_name, models, day0, wind, times):
    """Step-by-step forecast of every day-0 arrow. times: [t0, t1, ..., tn], t0 = end of day 0.

    Returns a list of steps; each step has start positions (x, y), velocity v (km/day) and hours.
    """
    x, y = moved(day0["x"], day0["y"], day0["v"], day0["hours"])
    return forecast_from(model_name, models, x, y, day0["v"], (day0["t0"], day0["t1"]), wind, times)


def forecast_from(model_name, models, x, y, v0, prev, wind, times):
    """Step-by-step forecast from start positions (x, y) at times[0], with drift v0 (km/day) measured over the
    window prev = (t_start, t_end) just before. Same return value as forecast()."""
    full, wind_only = models
    v_prev, day0 = v0, {"v": v0}
    steps = []
    for t0, t1 in zip(times, times[1:]):
        w_now, w_prev = wind.mean(x, y, t0, t1), wind.mean(x, y, *prev)
        v = {"full model": lambda: full.predict(v_prev, w_now, w_prev),
             "wind only": lambda: wind_only.predict(v_prev, w_now, w_prev),
             "persistence": lambda: day0["v"]}[model_name]()
        h = (t1 - t0).total_seconds() / 3600
        steps.append(dict(x=x, y=y, v=v, hours=h, t0=t0, t1=t1))
        x, y = moved(x, y, v, h)
        v_prev, prev = v, (t0, t1)
    return steps


def match(px, py, truth):
    """Index of the truth arrow starting nearest to each (px, py), or -1 if none within MATCH_KM."""
    d, idx = cKDTree(np.column_stack([truth["x"], truth["y"]])).query(np.column_stack([px, py]))
    return np.where(d <= MATCH_KM * 1e3, idx, -1)


def score(step, truth):
    """Displacement errors (km over the step) for forecast arrows matched to truth arrows."""
    idx = match(step["x"], step["y"], truth)
    ok = idx >= 0
    return vector_errors(step["v"][ok] * step["hours"] / 24, truth["v"][idx[ok]] * truth["hours"] / 24)


class OsiSafDrift:
    """OSI SAF 48 h drift fields (km/day, map frame) of 2026, interpolated in space; piecewise constant in time."""

    def __init__(self, first: datetime, last: datetime):
        self.windows = []
        d = datetime(2026, 10, 27, 12)
        while d < last:
            z = np.load(OSI_REGION / f"{d:%Y%m%d}.npz")
            ok = z["flag"] == 30
            hrs = 48 + np.nan_to_num(z["dt1"] - z["dt0"]) / 3600
            v = (z["dX"] + 1j * z["dY"]) / (hrs / 24)
            px, py = TO_MAP.transform(z["lon"][ok], z["lat"][ok])
            pts = np.column_stack([px, py])
            self.windows.append((d, d + timedelta(days=2), LinearNDInterpolator(pts, v[ok]), NearestNDInterpolator(pts, v[ok])))
            d += timedelta(days=2)
        if self.windows[0][0] > first:
            raise ValueError("OSI SAF windows start after the forecast start")

    def velocity(self, x, y, t: datetime):
        for t0, t1, lin, near in self.windows:
            if t0 <= t < t1:
                v = lin(x, y)
                return np.where(np.isfinite(v), v, near(x, y))
        raise ValueError(f"no OSI SAF window for {t}")

    def trajectory(self, x, y, times, dt_h=1.0):
        """Positions at each of `times`, moving hour by hour with the OSI SAF drift."""
        out, t = [(x, y)], times[0]
        for t_next in times[1:]:
            while t < t_next:
                x, y = moved(x, y, self.velocity(x, y, t), dt_h)
                t += timedelta(hours=dt_h)
            out.append((x, y))
        return out


def main():
    models = load_models()
    scenes, pairs = {}, []
    for a, b in zip(CHAIN, CHAIN[1:]):
        for n in (a, b):
            scenes.setdefault(n, load_gcov(DATA / f"{n}.h5", pols=["hh"]))
        pairs.append(track_pair(scenes[a], scenes[b], a, b))
        scenes.pop(a)                                           # keep at most two scenes in memory
        print(f"NISAR {start_time(a):%d %b %H:%M} -> {start_time(b):%d %b %H:%M} ({pairs[-1]['hours']:.1f} h): "
              f"{len(pairs[-1]['x'])} arrows, median {np.median(np.abs(pairs[-1]['v'])):.1f} km/day")
    scenes.clear()
    day0, day1, day2 = pairs
    t_start = day0["t1"]
    week = [t_start + timedelta(days=k) for k in range(DAYS + 1)]
    wind = Wind(day0["x"], day0["y"], f"{day0['t0'] - timedelta(days=1):%Y-%m-%d}", f"{week[-1] + timedelta(days=1):%Y-%m-%d}")

    # --- 1 and 2 days ahead at the NISAR pass times, checked against NISAR
    nisar_times = [day1["t0"], day1["t1"], day2["t1"]]
    short = {m: forecast(m, models, day0, wind, nisar_times) for m in METHODS}
    result = {"day 1, 1-day forecast": {m: score(short[m][0], day1) for m in METHODS},
              "day 2, 2-day forecast": {m: score(short[m][1], day2) for m in METHODS}}
    for task, res in result.items():
        print(f"\n{task} vs NISAR  (displacement error over the day, km)")
        for m, s in res.items():
            print(f"  {m:12s} n={s['n']:4d}  RMSE {s['rmse']:5.2f}   observed mean {s['mean_obs_speed']:5.2f}   "
                  f"length bias {s['speed_bias']:+5.2f}   median direction error {s['median_abs_angle_deg']:4.0f}°")

    # --- 7 days ahead, 24 h steps, checked against OSI SAF trajectories
    long = {m: forecast(m, models, day0, wind, week) for m in METHODS}
    x0, y0 = long["full model"][0]["x"], long["full model"][0]["y"]
    osi = OsiSafDrift(week[0], week[-1]).trajectory(x0, y0, week)
    pos = {m: [(s["x"], s["y"]) for s in long[m]] + [moved(long[m][-1]["x"], long[m][-1]["y"], long[m][-1]["v"], 24)]
           for m in METHODS}
    result["7-day position error vs OSI SAF (median km)"] = {
        m: [round(float(np.median(np.hypot(pos[m][k][0] - osi[k][0], pos[m][k][1] - osi[k][1]))) / 1e3, 1)
            for k in range(1, DAYS + 1)] for m in METHODS}
    result["7-day distance travelled, OSI SAF (median km)"] = [
        round(float(np.median(np.hypot(osi[k][0] - x0, osi[k][1] - y0))) / 1e3, 1) for k in range(1, DAYS + 1)]
    w_mean = [np.abs(wind.mean(s["x"], s["y"], s["t0"], s["t1"])).mean() for s in long["full model"]]
    print(f"\n7-day forecast from {week[0]:%d %b %H:%M} UTC, {len(x0)} pieces of ice. Median position error vs OSI SAF (km):")
    print("  day          " + "".join(f"{k:7d}" for k in range(1, DAYS + 1)))
    print("  ERA5 wind    " + "".join(f"{w:6.1f} " for w in w_mean) + " m/s")
    print("  OSI SAF dist " + "".join(f"{v:7.1f}" for v in result["7-day distance travelled, OSI SAF (median km)"]))
    for m in METHODS:
        print(f"  {m:12s} " + "".join(f"{v:7.1f}" for v in result["7-day position error vs OSI SAF (median km)"][m]))

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    figure_short(day1, day2, short["full model"], result)
    figure_week(day0, pos, osi, week)


def background(ax, name, grid):
    sc = load_gcov(DATA / f"{name}.h5", pols=["hh"])
    x, y = grid
    hh = to_db(on_grid(sc, "hh", x, y))
    ax.imshow(hh[::2, ::2], cmap="gray", extent=[x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3],
              vmin=np.nanpercentile(hh[::16, ::16], 2), vmax=np.nanpercentile(hh[::16, ::16], 98))


def figure_short(day1, day2, steps, result):
    """All forecast arrows and all NISAR arrows for day 1 and day 2, same map extent in both panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    q = dict(angles="xy", scale_units="xy", scale=1, width=0.0022)
    pts = []
    for s, t in zip(steps, (day1, day2)):
        for ax_x, ax_y, v, h in ((s["x"], s["y"], s["v"], s["hours"]), (t["x"], t["y"], t["v"], t["hours"])):
            ex, ey = moved(ax_x, ax_y, v, h)
            pts += [np.column_stack([ax_x, ax_y]), np.column_stack([ex, ey])]
    pts = np.concatenate(pts) / 1e3
    lim = (pts[:, 0].min() - 10, pts[:, 0].max() + 10, pts[:, 1].min() - 10, pts[:, 1].max() + 10)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.8), layout="constrained")
    for ax, s, truth, task, label in ((axes[0], steps[0], day1, "day 1, 1-day forecast", "Day 1"),
                                      (axes[1], steps[1], day2, "day 2, 2-day forecast", "Day 2")):
        background(ax, truth["start"], truth["grid"])
        dv = s["v"] * s["hours"] / 24
        ax.quiver(s["x"] / 1e3, s["y"] / 1e3, dv.real, dv.imag, color=COLORS["full model"], **q)
        tv = truth["v"] * truth["hours"] / 24
        ax.quiver(truth["x"] / 1e3, truth["y"] / 1e3, tv.real, tv.imag, color=COLORS["NISAR"], **q)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
        r = result[task]
        ax.set_title(f"{label}: {truth['t0']:%d %b %H:%M} -> {truth['t1']:%d %b %H:%M} UTC "
                     f"({'1' if label == 'Day 1' else '2'}-day forecast)\n"
                     f"{len(s['x'])} forecast arrows, {len(truth['x'])} NISAR arrows; where both exist ({r['full model']['n']}): "
                     f"error model {r['full model']['rmse']:.1f} km, persistence {r['persistence']['rmse']:.1f} km", fontsize=11)
        ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
    axes[0].legend(handles=[Line2D([], [], color=COLORS["NISAR"], lw=3, label="NISAR observed"),
                            Line2D([], [], color=COLORS["full model"], lw=3, label="model forecast")],
                   loc="lower left", framealpha=0.9)
    fig.suptitle("Ice drift forecast from the 27 -> 28 Oct NISAR drift + ERA5 wind (arrows = displacement at map scale). "
                 "Background: radar image at the start of the day; NISAR arrows exist only where two passes overlap.",
                 fontsize=11)
    plt.savefig(FIGS / "chukchi_drift_forecast.png", dpi=100); plt.close()


def figure_week(day0, pos, osi, week):
    """7-day trajectories of a spread-out subset of the ice: model, OSI SAF and persistence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x0, y0 = pos["full model"][0]
    keys = np.round(x0 / 25e3).astype(int) * 1000 + np.round(y0 / 25e3).astype(int)
    pick = np.unique(keys, return_index=True)[1]                 # one piece of ice per 25 km cell
    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    background(ax, CHAIN[1], common_grid([load_gcov(DATA / f"{n}.h5", pols=["hh"]) for n in CHAIN[:2]]))
    for name, path, style in (("persistence", pos["persistence"], dict(ls=":", lw=1.4)),
                              ("OSI SAF", osi, dict(ls="--", lw=1.6)),
                              ("full model", pos["full model"], dict(ls="-", lw=1.8))):
        px = np.array([p[0][pick] for p in path]) / 1e3
        py = np.array([p[1][pick] for p in path]) / 1e3
        ax.plot(px, py, color=COLORS[name], **style)
        ax.plot(px[1:], py[1:], "o", color=COLORS[name], ms=3.5)
    ax.plot(x0[pick] / 1e3, y0[pick] / 1e3, "o", color="white", mec="black", ms=6, zorder=5)
    end = np.array([p[pick] for p in pos["full model"][-1]]) / 1e3
    allp = np.concatenate([np.array([[p[0][pick], p[1][pick]] for p in pth]).transpose(0, 2, 1).reshape(-1, 2)
                           for pth in (pos["full model"], osi, pos["persistence"])]) / 1e3
    ax.set_xlim(allp[:, 0].min() - 15, allp[:, 0].max() + 15); ax.set_ylim(allp[:, 1].min() - 15, allp[:, 1].max() + 15)
    ax.annotate("day 7", end[:, np.argmax(end[1])], textcoords="offset points", xytext=(6, 6), fontsize=10,
                color=COLORS["full model"])
    ax.legend(handles=[Line2D([], [], color="white", marker="o", mec="black", ls="", label=f"start {week[0]:%d %b %H:%M} UTC"),
                       Line2D([], [], color=COLORS["full model"], lw=2, marker="o", ms=4, label="model forecast (dot = each day)"),
                       Line2D([], [], color=COLORS["OSI SAF"], lw=2, ls="--", marker="o", ms=4, label="OSI SAF observed drift (62.5 km)"),
                       Line2D([], [], color=COLORS["persistence"], lw=2, ls=":", marker="o", ms=4, label="persistence (keep 27->28 Oct drift)")],
              loc="lower right", framealpha=0.9)
    ax.set_title(f"7-day ice drift forecast, {week[0]:%d %b} -> {week[-1]:%d %b %Y}: {len(pick)} of {len(x0)} pieces of ice shown\n"
                 "Start: end of each 27 -> 28 Oct NISAR arrow. Wind: ERA5 (perfect-forecast assumption)", fontsize=11)
    ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
    plt.savefig(FIGS / "chukchi_drift_forecast_7day.png", dpi=100); plt.close()


if __name__ == "__main__":
    main()
