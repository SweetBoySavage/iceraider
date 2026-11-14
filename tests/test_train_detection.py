import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_detection import ICE, WATER, DarkIsWater, FixedRule, candidates, held_out_predictions, scores

NAMES = ["hh", "vv", "ratio", "incidence", "hh_texture", "ratio_texture", "hh_smooth", "ratio_smooth"]


def fake_table(n_scenes=6, per=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.r_[np.zeros(per // 2, int), np.ones(per // 2, int)], n_scenes)
    X = rng.normal(size=(len(y), len(NAMES)))
    X[:, 2] = np.where(y == WATER, -5, -1) + rng.normal(0, 0.5, len(y))       # ratio separates the classes
    X[:, 7] = X[:, 2]
    return X, y, np.repeat(np.arange(n_scenes), per)


def test_fixed_rule():
    X = np.array([[-5.0], [-1.0], [-3.01]])
    assert FixedRule(0).fit(X, None).predict(X).tolist() == [WATER, ICE, WATER]


def test_every_candidate_learns_the_easy_case_on_held_out_scenes():
    X, y, scene = fake_table()
    for name, (cols, model) in candidates(NAMES).items():
        pred = held_out_predictions(model, X[:, cols], y, scene, folds=3)
        assert np.mean(pred == y) > 0.95, name


def test_scores_report_the_special_cases():
    y = np.array([0, 0, 1, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 1, 0])
    scene = np.array([0, 0, 0, 0, 1, 1])
    laser = np.array([False, False, True, False, False, False])
    s = scores(y, pred, laser, scene, ["edge", "all ice"])
    assert s["water_found"] == 50.0 and s["ice_found"] == 75.0 and s["balanced"] == 62.5
    assert s["ice_in_all_ice_scenes"] == 50.0 and s["laser_confirmed_ice_found"] == 100.0 and s["worst_scene"] == 50.0
    assert np.isnan(s["water_in_all_water_scenes"])


def test_very_dark_pixels_are_water_whatever_the_base_model_says():
    X = np.array([[-1.0, -15.0], [-1.0, -30.0], [-5.0, -15.0]])       # columns: ratio, hh_smooth
    assert DarkIsWater(FixedRule(0)).fit(X, None).predict(X).tolist() == [ICE, WATER, WATER]
