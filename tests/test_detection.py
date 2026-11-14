import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.detection import FEATURES, features, local_std, scene_time


def test_local_std_is_zero_on_flat_and_positive_on_noise():
    assert np.allclose(local_std(np.full((30, 30), 5.0), 9), 0)
    assert local_std(np.random.default_rng(0).normal(size=(60, 60)), 9)[30, 30] > 0.5


def test_features_have_all_names_and_mask_out_no_data_and_borders():
    rng = np.random.default_rng(0)
    hh = rng.gamma(39, 0.02 / 39, (80, 80))
    vv = rng.gamma(39, 0.06 / 39, (80, 80))               # VV three times brighter: about -4.8 dB ratio
    hh[:, :20] = np.nan
    scene = SimpleNamespace(power={"hh": hh, "vv": vv}, looks={"hh": 39.0, "vv": 39.0}, incidence=np.full((80, 80), 40.0))
    feats, usable = features(scene)
    assert sorted(feats) == sorted(FEATURES)
    assert not usable[:, :25].any() and usable[40, 60]
    assert abs(feats["ratio"][usable].mean() - 10 * np.log10(1 / 3)) < 0.3
    assert all(np.isfinite(v[usable]).all() for v in feats.values())


def test_scene_time_from_name():
    t = scene_time("NISAR_L2_PR_GCOV_034_049_A_039_0505_SVSH_A_20261027T141811_20261027T141848_P05036_N_F_J_001")
    assert (t.day, t.hour, t.minute) == (27, 14, 18)
