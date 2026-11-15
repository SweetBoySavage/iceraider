"""Drift forecast for the three ice-edge test scenes of evaluate_detection.py (figures/detection_test.png).

    python scripts/forecast_test_scenes.py --download   # once: the 9 NISAR scenes below (about 3.1 GB)
    python scripts/forecast_test_scenes.py              # forecast + check, figure and json

NISAR does not see these places (69-73 N) every day, so the steps are about 48 h instead of 24 h, which is
also the step the drift model was trained on (OSI SAF 48 h windows). For each test scene T:

    before  B  (about 48 h earlier, same pass direction)   B -> T  = previous drift (the model input)
    test    T  (the scene in detection_test.png)
    after   A  (about 48 h later)                            T -> A  = what really happened (the check)

The forecast starts at the end of every B -> T arrow and predicts T -> A with
    V = a * V(B->T) + b * W(T->A) + c * W(B->T) + d      (ERA5 wind, i.e. a perfect wind forecast)
and is compared with the NISAR T -> A arrows that start within MATCH_KM, against persistence and wind only.

Channel: each pair is tracked on a channel that has signal in both images, VV first (the A passes are
from 1-2 Nov, when NISAR recorded VV only here), else HH (the 29 Oct 08:03 pass has an empty VV channel).
Similarity threshold 0.5, as in run_chukchi.py. Over 48 h at the ice edge the texture changes a lot, so few
or no patches reach it (for the 30 Oct -> 1 Nov pair: none; 0.3 would give 10, 0.2 would give 36).
For the 31 Oct 08:19 scene the 48 h passes share almost no area, so its A pass is the ascending one
55 h later (31 % shared by all three images).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.forecast_chukchi import COLORS, METHODS, Wind, forecast, score, track_pair  # noqa: E402
from scripts.nisar_loader import DeadChannelError, load_gcov                # noqa: E402
from scripts.run_chukchi import DATA, FIGS, TRACK, start_time              # noqa: E402
from scripts.train_drift_model import load_models                           # noqa: E402

POLS = ("vv", "hh")                      # tried in this order
TRACK_48H = TRACK                        # same settings as run_chukchi.py (similarity >= 0.5)
CHAINS = [  # (before, test, after)
    ("NISAR_L2_PR_GCOV_034_062_A_041_0505_SVSH_A_20261028T115751_20261028T115822_P05036_N_F_J_001",
     "NISAR_L2_PR_GCOV_034_091_A_040_0505_SVSH_A_20261030T121356_20261030T121431_P05036_N_F_J_001",
     "NISAR_L2_PR_GCOV_034_120_A_040_0005_NASV_A_20261101T123035_20261101T123110_P05036_N_F_J_001"),
    ("NISAR_L2_PR_GCOV_034_074_D_053_0505_SHSV_A_20261029T080307_20261029T080344_P05036_N_F_J_001",
     "NISAR_L2_PR_GCOV_034_103_D_053_0505_SVSH_A_20261031T081946_20261031T082013_P05036_N_P_J_001",
     "NISAR_L2_PR_GCOV_034_136_A_037_0005_NASV_A_20261102T150657_20261102T150734_P05036_N_F_J_001"),
    ("NISAR_L2_PR_GCOV_034_077_A_041_0505_SVSH_A_20261029T125607_20261029T125638_P05036_N_F_J_001",
     "NISAR_L2_PR_GCOV_034_106_A_040_0505_SVSH_A_20261031T131212_20261031T131247_P05036_N_F_J_001",
     "NISAR_L2_PR_GCOV_034_135_A_040_0005_NASV_A_20261102T132851_20261102T132926_P05036_N_F_J_001"),
]
OUT_JSON = ROOT / "models" / "drift_forecast_test_scenes.json"


def download():
    import asf_search as asf
    from scripts.download_scenes import find
    names = [n for chain in CHAINS for n in chain]
    session = asf.ASFSession()                                  # reads ~/.netrc
    for p in find(names):
        if not (DATA / p.properties["fileName"]).exists():
            print("downloading", p.properties["fileName"], flush=True)
            p.download(path=str(DATA), session=session, fileType=asf.FileDownloadType.DEFAULT_FILE)
    print("all", len(names), "scenes present")


def load_pair(a, b):
    """Both scenes with the first channel in POLS that has signal in both."""
    for pol in POLS:
        try:
            return pol, load_gcov(DATA / f"{a}.h5", pols=[pol]), load_gcov(DATA / f"{b}.h5", pols=[pol])
        except (DeadChannelError, KeyError, ValueError):
            continue
    raise ValueError(f"no channel with signal in both {a} and {b}")


def run_chain(before, test, after, models):
    pol0, sb, st0 = load_pair(before, test)
    day0 = track_pair(sb, st0, before, test, pol=pol0, track_kw=TRACK_48H)
    del sb, st0
    pol1, st, sa = load_pair(test, after)
    truth = track_pair(st, sa, test, after, pol=pol1, track_kw=TRACK_48H)
    del sa
    print(f"\n{start_time(test):%d %b %H:%M} test scene")
    for name, p, pol in (("before -> test", day0, pol0), ("test -> after", truth, pol1)):
        sp = np.abs(p["v"])
        print(f"  NISAR {name:15s} {pol.upper()} {p['hours']:5.1f} h  {len(sp):4d} arrows"
              + (f", median {np.median(sp):.1f} km/day" if len(sp) else ""))
    res = {"before": before, "test": test, "after": after, "channels": [pol0, pol1], "arrows_before_test": len(day0["x"]),
           "arrows_test_after": len(truth["x"]), "scores": {}}
    if not len(day0["x"]):
        print("  no drift tracked before the test scene: nothing to forecast from")
        return res, None
    wind = Wind(day0["x"], day0["y"], f"{day0['t0']:%Y-%m-%d}", f"{truth['t1']:%Y-%m-%d}")
    steps = {m: forecast(m, models, day0, wind, [truth["t0"], truth["t1"]])[0] for m in METHODS}
    if not len(truth["x"]):
        print("  forecast made, but no drift tracked after the test scene: not checked")
        return res, dict(steps=steps, truth=truth)
    for m in METHODS:
        s = score(steps[m], truth)
        res["scores"][m] = s
        print(f"  {m:12s} n={s['n']:4d}  error {s['rmse']:5.1f} km   observed mean {s['mean_obs_speed']:5.1f} km"
              f"   length bias {s['speed_bias']:+5.1f} km   median direction error {s['median_abs_angle_deg']:4.0f}°")
    return res, dict(steps=steps, truth=truth)


def figure(panels):
    """Same scenes and same view as column 1 of figures/detection_test.png (evaluate_detection.figure): same
    order, full scene, HH in dB on -26..-11, masked to detection.features' usable pixels, same axes in km.
    Forecast (orange) and NISAR (blue) arrows on top; dashed box = area shared with the 'after' pass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from scripts import detection

    by_test = {res["test"]: (res, p) for res, p in panels}
    skipped = set(json.loads((ROOT / "models" / "test_result.json").read_text())["skipped_by_quality_gate"])
    order = [s["name"] for s in json.loads((ROOT / "config" / "dataset.json").read_text())["test"]
             if s["type"] == "edge" and s["name"] not in skipped][:3]          # the rule of evaluate_detection.figure
    assert set(order) == set(by_test), "test scenes differ from detection_test.png"
    fig, axes = plt.subplots(1, len(order), figsize=(6.4 * len(order), 7.2), squeeze=False)
    q = dict(angles="xy", scale_units="xy", scale=1, width=0.004)
    for ax, name in zip(axes[0], order):
        res, p = by_test[name]
        scene = load_gcov(DATA / f"{name}.h5")
        _, usable = detection.features(scene)
        extent = [scene.x[0] / 1e3, scene.x[-1] / 1e3, scene.y[-1] / 1e3, scene.y[0] / 1e3]
        hh = np.where(usable, 10 * np.log10(np.maximum(scene.power["hh"], 1e-4)), np.nan)
        ax.imshow(hh[::2, ::2], cmap="gray", vmin=-26, vmax=-11, extent=extent)
        ax.set_facecolor("#dddddd"); ax.set_xlabel("km"); ax.set_ylabel("km")
        del scene, hh, usable
        title = f"NISAR HH, {detection.scene_time(name):%d %b %H:%M} UTC"
        if p is None:
            ax.set_title(f"{title}\nno drift tracked before: no forecast", fontsize=10)
            continue
        truth, step = p["truth"], p["steps"]["full model"]
        gx, gy = truth["grid"]
        ax.add_patch(Rectangle((gx[0] / 1e3, gy[-1] / 1e3), (gx[-1] - gx[0]) / 1e3, (gy[0] - gy[-1]) / 1e3,
                               fill=False, ls="--", lw=1.2, ec="#ffd166"))
        for arr, color in ((step, COLORS["full model"]), (truth, COLORS["NISAR"])):
            d = arr["v"] * arr["hours"] / 24
            ax.quiver(arr["x"] / 1e3, arr["y"] / 1e3, d.real, d.imag, color=color, **q)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        s = res["scores"].get("full model")
        score = ("no NISAR arrows after: not checked" if s is None else
                 f"error model {s['rmse']:.1f} km, persistence {res['scores']['persistence']['rmse']:.1f} km ({s['n']} arrows)"
                 if s["n"] else "no forecast arrow meets a NISAR arrow")
        ax.set_title(f"{title}, forecast {truth['hours']:.0f} h ahead\n{len(step['x'])} forecast, {len(truth['x'])} NISAR arrows; {score}",
                     fontsize=10)
    axes[0][0].legend(handles=[Line2D([], [], color=COLORS["NISAR"], lw=3, label="NISAR observed (after pass)"),
                               Line2D([], [], color=COLORS["full model"], lw=3, label="model forecast"),
                               Line2D([], [], color="#ffd166", lw=1.5, ls="--", label="area shared with after pass")],
                      loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("Drift forecast on the test scenes of detection_test.png (same scenes and view as its first column). "
                 "Arrows = displacement at map scale; similarity >= 0.5", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGS / "detection_test_drift_forecast.png", dpi=80); plt.close()


def main():
    models = load_models()
    out = [run_chain(*chain, models) for chain in CHAINS]
    OUT_JSON.write_text(json.dumps([r for r, _ in out], indent=2))
    figure(out)
    print(f"\nwrote {OUT_JSON.name} and figures/detection_test_drift_forecast.png")


if __name__ == "__main__":
    download() if "--download" in sys.argv else main()
