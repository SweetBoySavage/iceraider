"""Run the FROZEN ice / open-water model on the locked test scenes. Run this once.

    python scripts/evaluate_detection.py          # add --no-db to skip MongoDB

Refuses to run if no model is frozen. Writes models/test_result.json and figures/detection_test.png.

Map of any downloaded scene (sea ice / open water / land), without touching the test result:

    python scripts/evaluate_detection.py map SCENE_NAME      # writes figures/map_<start time>.png
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detection, land_mask, mongo_store, reference         # noqa: E402
from scripts.nisar_loader import load_gcov                               # noqa: E402
from scripts.train_detection import ICE, WATER, load_table, scores       # noqa: E402

MODELS, FIGS = ROOT / "models", ROOT / "figures"
WATER_C, ICE_C, UNSURE_C, LAND_C = "#1f4e79", "#ffffff", "#9ecae1", "#8c6d46"


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


SEA_ICE, OPEN_WATER, LAND, NO_DATA = 0, 1, 2, 255


def classify_scene(frozen, scene, margin_m: float | None = None) -> np.ndarray:
    """Map of SEA_ICE / OPEN_WATER / LAND / NO_DATA (uint8). Land comes from the coastline, never from the model.

    The model first answers on the untouched radar data, exactly as in training; land is painted over afterwards.
    `margin_m=None` takes the safety margin around land from LAND_MARGIN_M (.env).
    """
    water, usable = predict_scene(frozen, scene)
    out = np.full(usable.shape, NO_DATA, "uint8")
    out[usable] = np.where(water[usable], OPEN_WATER, SEA_ICE)
    land = land_mask.land_mask(scene.x, scene.y, scene.epsg, margin_m)
    out[land & np.isfinite(scene.power["hh"])] = LAND
    return out


def draw_classes(ax, classes, extent, step: int = 2):
    """Draw a classify_scene map: white = sea ice, blue = open water, brown = land, background = no data."""
    from matplotlib.colors import ListedColormap
    shown = np.where(classes == NO_DATA, np.nan, classes.astype("float32"))
    ax.imshow(shown[::step, ::step], cmap=ListedColormap([ICE_C, WATER_C, LAND_C]), vmin=0, vmax=2, extent=extent, interpolation="nearest")
    ax.set_title("Our model: white = sea ice, blue = open water, brown = land")


def map_figure(frozen, name: str) -> Path:
    """Radar image next to the three-class map of one scene. Returns the path of the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scene = load_gcov(detection.DATA / f"{name}.h5")
    margin = land_mask.margin_from_env()
    classes = classify_scene(frozen, scene, margin)
    extent = [scene.x[0] / 1e3, scene.x[-1] / 1e3, scene.y[-1] / 1e3, scene.y[0] / 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(10 * np.log10(np.maximum(scene.power["hh"], 1e-4))[::2, ::2], cmap="gray", vmin=-26, vmax=-11, extent=extent)
    axes[0].set_title(f"NISAR HH, {detection.scene_time(name):%d %b %H:%M} UTC")
    draw_classes(axes[1], classes, extent)
    axes[1].set_title(axes[1].get_title() + f"\n(land includes a {margin:g} m safety margin)")
    for ax in axes:
        ax.set_facecolor("#dddddd"); ax.set_xlabel("km"); ax.set_ylabel("km")
    out = FIGS / f"map_{detection.scene_time(name):%Y%m%dT%H%M%S}.png"
    plt.tight_layout(); plt.savefig(out, dpi=70, bbox_inches="tight"); plt.close(fig)
    for cls, label in ((SEA_ICE, "sea ice"), (OPEN_WATER, "open water"), (LAND, "land")):
        print(f"{label:10s} {np.mean(classes[classes != NO_DATA] == cls):6.1%}")
    return out


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
        classes = classify_scene(frozen, scene)
        usable = classes != NO_DATA
        label, _ = detection.scene_labels(s["name"], scene)
        extent = [scene.x[0] / 1e3, scene.x[-1] / 1e3, scene.y[-1] / 1e3, scene.y[0] / 1e3]
        hh = np.where(usable, 10 * np.log10(np.maximum(scene.power["hh"], 1e-4)), np.nan)
        row[0].imshow(hh[::2, ::2], cmap="gray", vmin=-26, vmax=-11, extent=extent)
        row[0].set_title(f"NISAR HH, {detection.scene_time(s['name']):%d %b %H:%M} UTC")
        draw_classes(row[1], classes, extent)
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
    if len(sys.argv) == 3 and sys.argv[1] == "map":
        print("saved", map_figure(load_model(), sys.argv[2]))
    else:
        main(use_db="--no-db" not in sys.argv)
