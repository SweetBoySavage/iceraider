"""End-to-end run for the Chukchi Sea test area, NISAR only: load -> common grid -> track -> classify -> MongoDB.

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
from scripts.classify import dark_threshold, kmeans_classes, largest_clean_square, lee_filter, to_db  # noqa: E402
from scripts.download_scenes import SCENES, find                              # noqa: E402
from scripts.ice_tracking import highpass, to_velocity, track                 # noqa: E402
from scripts.nisar_loader import load_gcov                                    # noqa: E402

DATA, FIGS = ROOT / "data", ROOT / "figures"
EPSG, SPACING = 3413, 80.0
NISAR = [s for s in SCENES if s.startswith("NISAR")]
# (from, to, source label). Sentinel-1 scenes in SCENES are not used.
PAIRS = [(NISAR[0], NISAR[1], "NISAR tracking")]
TRACK = dict(patch=128, step=64, coarse=4, min_similarity=0.5)
CLASSIFY = (NISAR[0], NISAR[1])    # the tracking pair, 23.3 h apart
NISAR_ENL = 39.0


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
    return power, x, y


def classify_pair(power, tracks):
    """k-means on the same patch of ice in both NISAR scenes (the later one moved back by the measured drift)."""
    first, second = CLASSIFY
    drift = tracks[(first, second)]
    d_row, d_col = int(round(np.median(drift["d_row"]))), int(round(np.median(drift["d_col"])))
    later = {p: shift_back(v, d_row, d_col) for p, v in power[second].items()}
    valid = np.all([np.isfinite(v) for v in (*power[first].values(), *later.values())], axis=0)
    r, c, size = largest_clean_square(valid)
    win = (slice(r, r + size), slice(c, c + size))
    out = {}
    for name, bands in ((first, power[first]), (second, later)):
        chans = ("hh", "vv")
        db = [to_db(lee_filter(bands[p][win], enl=NISAR_ENL)) for p in chans]
        labels, centres = kmeans_classes(*db)
        thr = dark_threshold(centres)
        out[name] = dict(db=db, labels=labels, centres=centres, threshold=thr, chans=chans,
                         shares=[float(np.mean(labels == i)) for i in range(4)], dark=float(np.mean(db[0] < thr)))
    a, b = out[first]["db"][0], out[second]["db"][0]
    quarter = (a < np.percentile(a, 25)), (b < np.percentile(b, 25))
    stats = dict(window=(r, c, size), drift_px=(d_row, d_col),
                 hh_correlation_80m=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                 darkest_quarter_iou=float(np.sum(quarter[0] & quarter[1]) / np.sum(quarter[0] | quarter[1])))
    return out, stats


def save_figures(power, x, y, tracks, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    extent = [x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3]
    # one row per pair: before (image 1) and after (image 2), same zoom, same displacement arrows on both
    fig, axes = plt.subplots(len(tracks), 2, figsize=(16, 7.5 * len(tracks)), squeeze=False, layout="constrained")
    summaries = []
    for row, ((a, b), t) in zip(axes, tracks.items()):
        # displacement arrows at map scale: from the patch centre in image 1 to the matched centre in image 2
        x0, y0 = x[t["col"].astype(int)] / 1e3, y[t["row"].astype(int)] / 1e3
        dx_km, dy_km = t["d_col"] * SPACING / 1e3, -t["d_row"] * SPACING / 1e3    # rows grow southwards
        dist_km = np.hypot(dx_km, dy_km)
        _, _, speed = to_velocity(t["d_row"], t["d_col"], SPACING, hours_between(a, b))
        for ax, name, when in ((row[0], a, "Before"), (row[1], b, "After")):
            hh = to_db(power[name]["hh"])
            ax.imshow(hh[::2, ::2], cmap="gray", extent=extent,
                      vmin=np.nanpercentile(hh[::16, ::16], 2), vmax=np.nanpercentile(hh[::16, ::16], 98))
            q = ax.quiver(x0, y0, dx_km, dy_km, dist_km, cmap="plasma", angles="xy", scale_units="xy", scale=1,
                          width=0.0025)
            ax.set_title(f"{when}: NISAR HH, {start_time(name):%d %b %Y %H:%M} UTC")
            if len(speed):                          # zoom to the arrows (start and end) with a 20 km margin
                ax.set_xlim(min(x0.min(), (x0 + dx_km).min()) - 20, max(x0.max(), (x0 + dx_km).max()) + 20)
                ax.set_ylim(min(y0.min(), (y0 + dy_km).min()) - 20, max(y0.max(), (y0 + dy_km).max()) + 20)
            ax.set_xlabel("x (km, EPSG:3413)"); ax.set_ylabel("y (km)")
        fig.colorbar(q, ax=row, label="displacement (km)", shrink=0.8)
        med = f", median {np.median(dist_km):.1f} km ({np.median(speed):.1f} km/day)" if len(speed) else ""
        summaries.append(f"{t['source']}, {hours_between(a, b):.1f} h: {len(speed)} arrows{med}")
    fig.suptitle("\n".join(summaries) + "\nArrows start at a patch centre on the 'before' image and end where that "
                 "patch is found on the 'after' image (map scale).", fontsize=12)
    plt.savefig(FIGS / "chukchi_ice_motion.png", dpi=100); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for col, (name, c) in enumerate(classes.items()):
        sensor = "NISAR L-band"
        axes[0, col].imshow(c["db"][0], cmap="gray", vmin=np.percentile(c["db"][0], 2), vmax=np.percentile(c["db"][0], 98))
        axes[0, col].set_title(f"{sensor} HH (dB), {start_time(name):%d %b %H:%M} UTC")
        axes[1, col].imshow(c["labels"], cmap="viridis", interpolation="nearest")
        axes[1, col].set_title(f"k-means on {c['chans'][0].upper()}+{c['chans'][1].upper()} (0 = darkest)")
    plt.tight_layout(); plt.savefig(FIGS / "chukchi_classes.png", dpi=60); plt.close()


def store(db, x, y, tracks, classes, stats):
    products = {p.properties["sceneName"]: p for p in find()}
    scenes = []
    for name in NISAR:
        scenes.append(mongo_store.scene_doc(name, "NISAR", "GCOV", start_time(name), DATA / f"{name}.h5",
                                            products[name].geometry, ["HH", "VV"], "L", f"EPSG:{EPSG}"))
    for doc in scenes:
        db.scenes.replace_one({"name": doc["name"]}, doc, upsert=True)
    for name, c in classes.items():
        note = "Dark class = thin ice / leads, not confirmed open water. Thermal noise removed."
        doc = mongo_store.result_doc(name, "k-means k=4, then HH threshold between the two darkest centres",
                                     c["chans"], c["shares"], c["threshold"], c["dark"], stats["window"], note)
        db.results.replace_one({"scene": name, "method": doc["method"]}, doc, upsert=True)
    n = 0
    for (a, b), t in tracks.items():
        docs = mongo_store.motion_docs(t, x, y, EPSG, SPACING, start_time(a), hours_between(a, b), t["source"], a, b)
        n += mongo_store.replace_many(db.ice_motion, {"scene_from": a, "scene_to": b}, docs)
    return len(scenes), len(classes), n


def main(use_db: bool):
    power, x, y = load_everything()
    tracks = {}
    for a, b, source in PAIRS:
        t = track(texture(power[a]["hh"]), texture(power[b]["hh"]), **TRACK)
        t["source"] = source
        tracks[(a, b)] = t
        _, _, speed = to_velocity(t["d_row"], t["d_col"], SPACING, hours_between(a, b))
        print(f"{source:30s} {hours_between(a, b):5.1f} h  {len(speed):4d} vectors  median {np.median(speed):5.2f} km/day")
    classes, stats = classify_pair(power, tracks)
    for name, c in classes.items():
        print(f"{name[:5]} dark class {c['dark']:.1%} at HH < {c['threshold']:.1f} dB; class shares {np.round(c['shares'], 3)}")
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()})
    save_figures(power, x, y, tracks, classes)
    if use_db:
        print("MongoDB: scenes %d, results %d, ice_motion %d" % store(mongo_store.get_db(), x, y, tracks, classes, stats))


if __name__ == "__main__":
    main(use_db="--no-db" not in sys.argv)
