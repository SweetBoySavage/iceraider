"""Run the FROZEN ice / open-water model on the locked test scenes. Run this once.

    python scripts/evaluate_detection.py          # add --no-db to skip MongoDB

Refuses to run if no model is frozen. Writes models/test_result.json and figures/detection_test.png.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detection, mongo_store, reference                    # noqa: E402
from scripts.nisar_loader import load_gcov                               # noqa: E402
from scripts.train_detection import ICE, WATER, load_table, scores       # noqa: E402

MODELS, FIGS = ROOT / "models", ROOT / "figures"
WATER_C, ICE_C, UNSURE_C = "#1f4e79", "#ffffff", "#9ecae1"


def load_model():
    import joblib
    path = MODELS / "ice_water.joblib"
    if not path.exists():
        raise SystemExit("No frozen model. Run: python scripts/train_detection.py freeze NAME")
    return joblib.load(path)


def predict_scene(frozen, scene):
    """Water mask (True = open water) and usable mask for a loaded NISAR scene."""
    feats, usable = detection.features(scene)
    X = np.column_stack([feats[k][usable] for k in frozen["features"]])
    water = np.zeros(usable.shape, bool)
    water[usable] = frozen["model"].predict(X) == WATER
    return water, usable


def figure(frozen, scenes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    edge = [s for s in scenes if s["type"] == "edge"][:3]
    if not edge:
        return
    fig, axes = plt.subplots(len(edge), 3, figsize=(18, 6 * len(edge)), squeeze=False)
    for row, s in zip(axes, edge):
        scene = load_gcov(detection.DATA / f"{s['name']}.h5")
        water, usable = predict_scene(frozen, scene)
        label, _ = detection.scene_labels(s["name"], scene)
        extent = [scene.x[0] / 1e3, scene.x[-1] / 1e3, scene.y[-1] / 1e3, scene.y[0] / 1e3]
        hh = np.where(usable, 10 * np.log10(np.maximum(scene.power["hh"], 1e-4)), np.nan)
        row[0].imshow(hh[::2, ::2], cmap="gray", vmin=-26, vmax=-11, extent=extent)
        row[0].set_title(f"NISAR HH, {detection.scene_time(s['name']):%d %b %H:%M} UTC")
        row[1].imshow(np.where(usable, water, np.nan)[::2, ::2], cmap=ListedColormap([ICE_C, WATER_C]), vmin=0, vmax=1, extent=extent, interpolation="nearest")
        row[1].set_title("Our model: white = sea ice, blue = open water")
        ref = np.where(usable, np.select([label == ICE, label == WATER], [0, 2], 1), np.nan)
        row[2].imshow(ref[::2, ::2], cmap=ListedColormap([ICE_C, UNSURE_C, WATER_C]), vmin=0, vmax=2, extent=extent, interpolation="nearest")
        row[2].set_title("Reference labels: light blue = unsure (not scored)")
        for ax in row:
            ax.set_facecolor("#dddddd"); ax.set_xlabel("km"); ax.set_ylabel("km")
    plt.tight_layout()
    plt.savefig(FIGS / "detection_test.png", dpi=55)


def main(use_db: bool):
    frozen = load_model()
    table = detection.TABLES / "test.npz"
    if not table.exists():
        detection.build_table("test")
    X, y, laser, scene, names, types = load_table(table)
    cols = [names.index(f) for f in frozen["features"]]
    pred = frozen["model"].predict(X[:, cols])
    t = np.load(table, allow_pickle=False)
    scene_names, skipped = [str(n) for n in t["names"]], [str(n) for n in t["skipped"]]
    per_scene = {}
    for i in sorted(set(scene)):
        m = scene == i
        per_scene[scene_names[i]] = {"type": types[i], **{f"{k}_found": (round(100 * float(np.mean(pred[m & (y == c)] == c)), 1) if (m & (y == c)).any() else None)
                                                         for k, c in (("water", WATER), ("ice", ICE))}}
    result = {"model": frozen["name"], "features": frozen["features"], "evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "test_rows": int(len(y)), "test_scenes": int(len(set(scene))), "scores": scores(y, pred, laser, scene, types),
              "per_scene": per_scene, "skipped_by_quality_gate": skipped}
    (MODELS / "test_result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    scenes = json.loads((ROOT / "config" / "dataset.json").read_text())["test"]
    figure(frozen, [s for s in scenes if s["name"] not in skipped])
    if use_db:
        db = mongo_store.get_db()
        db.results.replace_one({"scene": "TEST SET 29-31 Oct 2026", "method": frozen["name"]},
                               {"scene": "TEST SET 29-31 Oct 2026", "method": frozen["name"], **result}, upsert=True)


if __name__ == "__main__":
    main(use_db="--no-db" not in sys.argv)
