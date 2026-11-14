"""Compare ice / open-water models on the TRAINING table only, then freeze one.

Run:  python scripts/train_detection.py            # compare candidates (leave-scenes-out)
      python scripts/train_detection.py freeze NAME  # fit NAME on all training data, save to models/

Fair comparison: every score comes from scenes the model did not see while fitting (whole scenes are
held out, 6 rounds). The test table is never read here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.classify import COPOL_RATIO_WATER_DB      # noqa: E402

TABLE, MODELS = ROOT / "data" / "tables" / "train.npz", ROOT / "models"
WATER, ICE = 0, 1


class FixedRule:
    """Baseline: open water where HH - VV < -3 dB. Nothing is learned."""

    def __init__(self, column: int, threshold: float = COPOL_RATIO_WATER_DB):
        self.column, self.threshold = column, threshold

    def fit(self, X, y, **_):
        return self

    def predict(self, X):
        return np.where(X[:, self.column] < self.threshold, WATER, ICE)


class DarkIsWater:
    """Wrap a model: pixels darker than any sea ice we have seen are open water (glass-calm water).

    Calm water sends almost nothing back in HH and VV, so their difference vanishes and the ratio fails.
    In training, ice was never below -27.6 dB in smoothed HH (0.1 % point); -28 dB is the cut.
    The LAST column given to this model must be hh_smooth; the wrapped model gets the other columns.
    """

    def __init__(self, base, threshold_db: float = -28.0):
        self.base, self.threshold_db = base, threshold_db

    def fit(self, X, y, **_):
        self.base.fit(X[:, :-1], y)
        return self

    def predict(self, X):
        return np.where(X[:, -1] < self.threshold_db, WATER, self.base.predict(X[:, :-1]))


def candidates(names: list[str]) -> dict:
    """name -> (columns used, model). Kept small on purpose: each one can be explained in a sentence."""
    col = {n: i for i, n in enumerate(names)}
    pick = lambda *ks: [col[k] for k in ks]                                      # noqa: E731
    logistic = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))   # noqa: E731
    boosted = lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1,                 # noqa: E731
                                                     class_weight="balanced", random_state=0)
    return {
        "rule: ratio < -3 dB": (pick("ratio"), FixedRule(0)),
        "rule on smoothed ratio": (pick("ratio_smooth"), FixedRule(0)),
        "logistic: ratio": (pick("ratio"), logistic()),
        "logistic: ratio + angle": (pick("ratio", "incidence"), logistic()),
        "logistic: ratio_smooth + angle": (pick("ratio_smooth", "incidence"), logistic()),
        "logistic: all but angle": (pick("hh", "vv", "ratio", "hh_texture", "ratio_texture", "hh_smooth", "ratio_smooth"), logistic()),
        "logistic: all": (list(range(len(names))), logistic()),
        "boosted trees: all but angle": (pick("hh", "vv", "ratio", "hh_texture", "ratio_texture", "hh_smooth", "ratio_smooth"), boosted()),
        "boosted trees: all": (list(range(len(names))), boosted()),
        "rule + dark": (pick("ratio", "hh_smooth"), DarkIsWater(FixedRule(0))),
        "logistic ratio + dark": (pick("ratio", "hh_smooth"), DarkIsWater(logistic())),
        "logistic ratio_smooth + dark": (pick("ratio_smooth", "hh_smooth"), DarkIsWater(logistic())),
        "trees (no angle) + dark": (pick("hh", "vv", "ratio", "hh_texture", "ratio_texture", "ratio_smooth", "hh_smooth"), DarkIsWater(boosted())),
        "trees (ratio, textures) + dark": (pick("ratio", "ratio_smooth", "hh_texture", "ratio_texture", "hh_smooth"), DarkIsWater(boosted())),
    }


def scores(y, pred, laser, scene, types) -> dict:
    """Balanced accuracy overall, plus the cases that matter: pure scenes and laser-confirmed ice."""
    water_found, ice_found = np.mean(pred[y == WATER] == WATER), np.mean(pred[y == ICE] == ICE)
    kind = np.array(types)[scene]
    per_scene = []
    for s in np.unique(scene):
        m = scene == s
        parts = [np.mean(pred[m & (y == c)] == c) for c in (WATER, ICE) if (m & (y == c)).any()]
        per_scene.append(np.mean(parts))
    out = {"balanced": (water_found + ice_found) / 2, "water_found": water_found, "ice_found": ice_found,
           "worst_scene": float(np.min(per_scene)),
           "ice_in_all_ice_scenes": np.mean(pred[kind == "all ice"] == ICE) if (kind == "all ice").any() else np.nan,
           "water_in_all_water_scenes": np.mean(pred[kind == "all water"] == WATER) if (kind == "all water").any() else np.nan,
           "laser_confirmed_ice_found": np.mean(pred[laser] == ICE) if laser.any() else np.nan}
    return {k: round(100 * float(v), 1) for k, v in out.items()}


def held_out_predictions(model, X, y, groups, folds=6):
    pred = np.empty(len(y), int)
    for fit_idx, test_idx in GroupKFold(n_splits=folds).split(X, y, groups):
        pred[test_idx] = model.fit(X[fit_idx], y[fit_idx]).predict(X[test_idx])
    return pred


def load_table(path=TABLE):
    t = np.load(path, allow_pickle=False)
    return t["X"], t["y"].astype(int), t["laser"], t["scene"], [str(n) for n in t["features"]], [str(n) for n in t["types"]]


def compare():
    X, y, laser, scene, names, types = load_table()
    print(f"training rows {len(y)} from {len(set(scene))} scenes | water {np.mean(y == WATER):.0%} | laser-confirmed ice {laser.sum()}")
    print(f"{'model':32s} balanced  water  ice   worst-scene  ice@all-ice  water@all-water  laser-ice")
    results = {}
    for name, (cols, model) in candidates(names).items():
        r = scores(y, held_out_predictions(model, X[:, cols], y, scene), laser, scene, types)
        results[name] = r
        print(f"{name:32s} {r['balanced']:7.1f} {r['water_found']:6.1f} {r['ice_found']:5.1f} {r['worst_scene']:11.1f} "
              f"{r['ice_in_all_ice_scenes']:11.1f} {r['water_in_all_water_scenes']:15.1f} {r['laser_confirmed_ice_found']:10.1f}", flush=True)
    return results


def freeze(name: str):
    import joblib
    X, y, laser, scene, names, types = load_table()
    cols, model = candidates(names)[name]
    held_out = scores(y, held_out_predictions(model, X[:, cols], y, scene), laser, scene, types)
    model.fit(X[:, cols], y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": [names[c] for c in cols], "name": name}, MODELS / "ice_water.joblib")
    (MODELS / "ice_water.json").write_text(json.dumps({
        "name": name, "features": [names[c] for c in cols], "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "training_rows": int(len(y)), "training_scenes": int(len(set(scene))), "held_out_scores_on_training_scenes": held_out}, indent=1))
    print("frozen:", name, held_out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "freeze":
        freeze(sys.argv[2])
    else:
        compare()
