"""End-to-end run for the Chukchi Sea test area: load -> common grid -> ice detection -> track -> classify -> MongoDB.

Run after scripts/download_scenes.py:   python scripts/run_chukchi.py
Add --no-db to skip MongoDB. Writes figures to figures/chukchi_*.png. Tracking is described in docs/ice_tracking.md.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mongo_store                                               # noqa: E402
from scripts.asf_loader import ENL_EW_GRDM, load_asf_item                     # noqa: E402
from scripts.classify import (dark_threshold, kmeans_classes, largest_clean_square, lee_filter, to_db,  # noqa: E402
                              water_from_copol_ratio)
from scripts.download_scenes import SCENES, find                              # noqa: E402
from scripts.ice_tracking import highpass, keep_on_ice, to_velocity, track                 # noqa: E402
from scripts.nisar_loader import load_gcov                                    # noqa: E402
from scripts.s1_geocode import sigma0_radar_geometry, warp_to_grid            # noqa: E402

DATA, FIGS = ROOT / "data", ROOT / "figures"
EPSG, SPACING = 3413, 80.0
NISAR = [s for s in SCENES if s.startswith("NISAR")]
S1 = [s for s in SCENES if s.startswith("S1")]
# (from, to, source label). Sentinel-1 27 -> 29 Oct is left out: 48 h is too long, the ice pattern changes.
PAIRS = [(NISAR[0], NISAR[1], "NISAR tracking"),
         (S1[0], S1[1], "Sentinel-1 tracking"),
         (NISAR[0], S1[1], "NISAR to Sentinel-1 tracking"),
         (S1[1], NISAR[1], "Sentinel-1 to NISAR tracking")]
TRACK = dict(patch=128, step=64, coarse=4, min_similarity=0.3)
CLASSIFY = (NISAR[0], S1[1])       # same day (27 Oct), 7.4 h apart


def start_time(name: str) -> datetime:
    return datetime.strptime(re.search(r"\d{8}T\d{6}", name).group(), "%Y%m%dT%H%M%S")


def hours_between(a: str, b: str) -> float:
    return (start_time(b) - start_time(a)).total_seconds() / 3600


def common_grid(scenes):
    """Pixel-centre coordinates of the area all NISAR scenes share (their grids are already aligned)."""
    x0, x1 = max(s.x[0] for s in scenes), min(s.x[-1] for s in scenes)
    y0, y1 = min(s.y[0] for s in scenes), max(s.y[-1] for s in scenes)
    return np.arange(x0, x1 + 1, SPACING), np.arange(y0, y1 - 1, -SPACING)


def on_grid(scene, pol, x, y):
    c0, r0 = int(round((x[0] - scene.x[0]) / SPACING)), int(round((scene.y[0] - y[0]) / SPACING))
    return scene.power[pol][r0:r0 + len(y), c0:c0 + len(x)]


def texture(power):
    """dB image with slow brightness changes removed; no-data stays NaN."""
    db, valid = to_db(power), np.isfinite(power)
    out = highpass(np.where(valid, db, np.nanmedian(db[valid])), sigma=25)
    out[~valid] = np.nan
    return out


def shift_back(img, d_row, d_col):
    """Move an image by (-d_row, -d_col) pixels: puts later ice back where it was earlier."""
    out = np.full_like(img, np.nan)
    h, w = img.shape
    out[max(0, -d_row):h - max(0, d_row), max(0, -d_col):w - max(0, d_col)] = \
        img[max(0, d_row):h - max(0, -d_row), max(0, d_col):w - max(0, -d_col)]
    return out


def load_everything():
    nisar = {n: load_gcov(DATA / f"{n}.h5") for n in NISAR}
    x, y = common_grid(list(nisar.values()))
    power = {n: {p: on_grid(s, p, x, y) for p in s.power} for n, s in nisar.items()}
    for name in S1:
        item = load_asf_item(name, DATA)
        power[name] = {p: warp_to_grid(*sigma0_radar_geometry(item, p), x, y, EPSG).astype("float64") for p in ("hh", "hv")}
    return power, x, y, {n: s.looks for n, s in nisar.items()}


def ice_mask(name, bands, looks):
    """Ice detection on one scene, before tracking: (ice, valid) bool arrays on the common grid.

    NISAR (HH+VV): open water where HH is more than 3 dB below VV, the rule verify_chukchi.py checks
    against AMSR2. Sentinel-1 (HH+HV, no VV): darkest k-means class on HH+HV. That rule is NOT validated;
    at C-band the dark class can also be smooth thin ice, so it may remove some ice as well as water.
    """
    nisar = name.startswith("NISAR")
    pols = ("hh", "vv") if nisar else ("hh", "hv")
    valid = np.all([np.isfinite(bands[p]) for p in pols], axis=0)
    db = {p: to_db(lee_filter(np.where(valid, bands[p], np.nanmedian(bands[p])),
                              enl=looks[name][p] if nisar else 4 * ENL_EW_GRDM)) for p in pols}
    if nisar:
        water = water_from_copol_ratio(db["hh"], db["vv"])
    else:
        _, centres = kmeans_classes(db["hh"][valid][::50], db["hv"][valid][::50])
        water = db["hh"] < dark_threshold(centres)
    return valid & ~water, valid


def classify_pair(power, tracks):
    """k-means on the same patch of ice in NISAR and in Sentinel-1 (moved back by the measured drift)."""
    n_name, s_name = CLASSIFY
    drift = tracks[(n_name, s_name)]
    d_row, d_col = int(round(np.median(drift["d_row"]))), int(round(np.median(drift["d_col"])))
    nis = power[n_name]
    s1 = {p: shift_back(v, d_row, d_col) for p, v in power[s_name].items()}
    valid = np.all([np.isfinite(v) for v in (*nis.values(), *s1.values())], axis=0)
    r, c, size = largest_clean_square(valid)
    win = (slice(r, r + size), slice(c, c + size))
    out = {}
    for name, bands, chans, enl in ((n_name, nis, ("hh", "vv"), 39.0), (s_name, s1, ("hh", "hv"), 4 * ENL_EW_GRDM)):
        db = [to_db(lee_filter(bands[p][win], enl=enl)) for p in chans]
        labels, centres = kmeans_classes(*db)
        thr = dark_threshold(centres)
        out[name] = dict(db=db, labels=labels, centres=centres, threshold=thr, chans=chans,
                         shares=[float(np.mean(labels == i)) for i in range(4)], dark=float(np.mean(db[0] < thr)))
    a, b = out[n_name]["db"][0], out[s_name]["db"][0]
    quarter = (a < np.percentile(a, 25)), (b < np.percentile(b, 25))
    stats = dict(window=(r, c, size), drift_px=(d_row, d_col),
                 hh_correlation_80m=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                 darkest_quarter_iou=float(np.sum(quarter[0] & quarter[1]) / np.sum(quarter[0] | quarter[1])))
    return out, stats


def save_figures(power, x, y, runs, classes):
    """runs: list of (row label, tracks, filter info or None); one figure row per run, one column per pair.
    Filter info = (ice masks per scene, removed arrows per pair): the row then shows image 1's detected
    water in blue and marks the start of every removed arrow with a black x."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    extent = [x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3]
    pairs = list(runs[0][1])
    fig, axes = plt.subplots(len(runs), len(pairs), figsize=(8 * len(pairs), 8 * len(runs)), squeeze=False)
    for j, (a, b) in enumerate(pairs):
        # one colour scale per pair (column), so the rows can be compared directly
        speeds = [to_velocity(t[(a, b)]["d_row"], t[(a, b)]["d_col"], SPACING, hours_between(a, b))[2] for _, t, _ in runs]
        allspeed = np.concatenate(speeds)
        norm = matplotlib.colors.Normalize(*(np.percentile(allspeed, (2, 98)) if len(allspeed) else (0, 1)))
        for i, (label, tracks, filt) in enumerate(runs):
            ax, t = axes[i, j], tracks[(a, b)]
            ax.imshow(to_db(power[a]["hh"][::4, ::4]), cmap="gray", extent=extent,
                      vmin=np.nanpercentile(to_db(power[a]["hh"][::16, ::16]), 2), vmax=np.nanpercentile(to_db(power[a]["hh"][::16, ::16]), 98))
            removed_note = ""
            if filt is not None:
                masks, removed = filt
                ice, valid = masks[a]                   # image 1's detected water in blue
                ax.imshow(np.ma.masked_where(~(valid & ~ice)[::4, ::4], np.ones(ice[::4, ::4].shape)),
                          cmap=matplotlib.colors.ListedColormap(["#1f6feb"]), vmin=0, vmax=1, alpha=0.45,
                          extent=extent, interpolation="nearest")
                r = removed[(a, b)]
                ax.plot(x[r["col"].astype(int)] / 1e3, y[r["row"].astype(int)] / 1e3, "kx", ms=5, mew=1.2)
                removed_note = f", {len(r['row'])} removed (x)"
            vx, vy, speed = to_velocity(t["d_row"], t["d_col"], SPACING, hours_between(a, b))
            q = ax.quiver(x[t["col"].astype(int)] / 1e3, y[t["row"].astype(int)] / 1e3, vx, vy, speed, cmap="plasma",
                          norm=norm, angles="xy", scale_units="xy", scale=0.02, width=0.003)
            plt.colorbar(q, ax=ax, label="km/day", shrink=0.6)
            med = f", median {np.median(speed):.1f} km/day" if len(speed) else ""
            ax.set_title(f"{label}: {t['source']}\n{start_time(a):%d %b %H:%M} -> {start_time(b):%d %b %H:%M} UTC, "
                         f"{len(speed)} vectors{med}{removed_note}")
            ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
    plt.tight_layout(); plt.savefig(FIGS / "chukchi_ice_motion.png", dpi=70); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for col, (name, c) in enumerate(classes.items()):
        sensor = "NISAR L-band" if name.startswith("NISAR") else "Sentinel-1 C-band"
        axes[0, col].imshow(c["db"][0], cmap="gray", vmin=np.percentile(c["db"][0], 2), vmax=np.percentile(c["db"][0], 98))
        axes[0, col].set_title(f"{sensor} HH (dB), {start_time(name):%d %b %H:%M} UTC")
        axes[1, col].imshow(c["labels"], cmap="viridis", interpolation="nearest")
        axes[1, col].set_title(f"k-means on {c['chans'][0].upper()}+{c['chans'][1].upper()} (0 = darkest)")
    plt.tight_layout(); plt.savefig(FIGS / "chukchi_classes.png", dpi=60); plt.close()


def store(db, x, y, tracks, classes, stats):
    products = {p.properties["sceneName"]: p for p in find()}
    scenes = []
    for name in SCENES:
        nisar = name.startswith("NISAR")
        scenes.append(mongo_store.scene_doc(
            name, "NISAR" if nisar else "Sentinel-1", "GCOV" if nisar else "GRD", start_time(name),
            DATA / (f"{name}.h5" if nisar else f"{name}.SAFE"), products[name].geometry,
            ["HH", "VV"] if nisar else ["HH", "HV"], "L" if nisar else "C",
            f"EPSG:{EPSG}" if nisar else "radar geometry (ground range), tie points in EPSG:4326"))
    for doc in scenes:
        db.scenes.replace_one({"name": doc["name"]}, doc, upsert=True)
    for name, c in classes.items():
        note = ("Dark class = thin ice / leads, not confirmed open water. "
                + ("Thermal noise removed." if name.startswith("NISAR") else "Thermal noise NOT removed; HV shows a sub-swath step."))
        doc = mongo_store.result_doc(name, "k-means k=4, then HH threshold between the two darkest centres",
                                     c["chans"], c["shares"], c["threshold"], c["dark"], stats["window"], note)
        db.results.replace_one({"scene": name, "method": doc["method"]}, doc, upsert=True)
    n = 0
    for (a, b), t in tracks.items():
        docs = mongo_store.motion_docs(t, x, y, EPSG, SPACING, start_time(a), hours_between(a, b), t["source"], a, b)
        n += mongo_store.replace_many(db.ice_motion, {"scene_from": a, "scene_to": b}, docs)
    return len(scenes), len(classes), n


def main(use_db: bool):
    power, x, y, looks = load_everything()

    # 1. ice detection on every scene used in a pair
    masks = {n: ice_mask(n, power[n], looks) for n in dict.fromkeys(n for a, b, _ in PAIRS for n in (a, b))}
    for name, (ice, valid) in masks.items():
        print(f"detection {name[:5]} {start_time(name):%d %b %H:%M}  open water {1 - ice[valid].mean():6.1%} of valid pixels")

    # 2. track on the radar texture, then drop arrows that start in water (image 1) or end in water (image 2)
    tracks, radar, removed = {}, {}, {}
    for a, b, source in PAIRS:
        t = track(texture(power[a]["hh"]), texture(power[b]["hh"]), **TRACK)
        t["source"] = source
        radar[(a, b)] = t
        tracks[(a, b)], keep = keep_on_ice(t, masks[a][0], masks[b][0])
        removed[(a, b)] = {k: (v[~keep] if isinstance(v, np.ndarray) else v) for k, v in t.items()}
        speeds = [to_velocity(o["d_row"], o["d_col"], SPACING, hours_between(a, b))[2] for o in (t, tracks[(a, b)])]
        print(f"{source:30s} {hours_between(a, b):5.1f} h  " + "  ".join(
            f"{label} {len(s):4d} vectors, median {np.median(s) if len(s) else float('nan'):5.2f} km/day"
            for label, s in zip(("all:", "on ice:"), speeds)))
    # classification lines up the two radar images with the median drift of all arrows
    classes, stats = classify_pair(power, radar)
    for name, c in classes.items():
        print(f"{name[:5]} dark class {c['dark']:.1%} at HH < {c['threshold']:.1f} dB; class shares {np.round(c['shares'], 3)}")
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()})
    save_figures(power, x, y, [("All arrows", radar, None), ("On ice", tracks, (masks, removed))], classes)
    if use_db:
        print("MongoDB: scenes %d, results %d, ice_motion %d" % store(mongo_store.get_db(), x, y, tracks, classes, stats))


if __name__ == "__main__":
    main(use_db="--no-db" not in sys.argv)
