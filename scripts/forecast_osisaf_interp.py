"""Drift forecast on a 10 km grid, started from OSI SAF drift INTERPOLATED from its 62.5 km cells.

    python scripts/forecast_osisaf_interp.py      # needs build_drift_training.py and train_drift_model.py

1. Starting drift V0: the OSI SAF map of 26 Oct 12:00 -> 28 Oct 12:00 UTC 2026 (Chukchi / western Beaufort box,
   nominal-quality cells), interpolated linearly to a 10 km grid inside the valid cells (nearest cell outside).
2. Wind: ERA5 10 m wind at the OSI SAF cell centres, averaged over each 48 h window and interpolated the same way.
3. Forecast with the trained model (models/drift_model.json), 48 h steps on the fixed grid, each step from the
   previous prediction:   V1 = a*V0 + b*W1 + c*W0 + d,   V2 = a*V1 + b*W2 + c*W1 + d, ...
   W is the wind that actually blew (a perfect wind forecast).
4. Check: the OSI SAF drift of each later window, interpolated to the same points; also scored at the OSI SAF
   cell centres. The model is linear and drift and wind are interpolated linearly, so the 10 km forecast equals
   the 62.5 km forecast interpolated afterwards: more points, not more information.

Starting from OSI SAF was checked against NISAR-tracked drift (Chukchi, 28 -> 29 Oct, 64 points with a NISAR arrow): error 1.2 km over
the day, vs 1.6 km when starting from the NISAR drift itself; see docs/drift_forecast.md.

Writes figures/osisaf_interp_grid_forecast.png and models/osisaf_interp_result.json.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_drift_training import BOX, NOMINAL, fetch_region           # noqa: E402
from scripts.drift_model import east_north_to_map, vector_errors              # noqa: E402
from scripts.forecast_chukchi import COLORS                                    # noqa: E402
from scripts.run_chukchi import FIGS                                          # noqa: E402
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


def forecast_grid(models):
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
    print(f"\nGrid: {inside.sum()} fine-grid points ({FINE_KM:.0f} km) from {cells.sum()} OSI SAF cells; "
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
    result.update(forecast_grid(models))
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} and figures/osisaf_interp_grid_forecast.png")


if __name__ == "__main__":
    main()
