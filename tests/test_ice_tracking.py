import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ice_tracking import keep_on_ice, similarity, to_velocity, track


def textured(shape, seed=0):
    return gaussian_filter(np.random.default_rng(seed).normal(size=shape), 2)


@pytest.mark.parametrize("move", [(3, -5), (40, 25), (-90, 60)])   # small, medium, larger than half a patch
def test_recovers_a_known_shift(move):
    base = textured((1100, 1100))
    img1 = base[150:950, 150:950]
    img2 = base[150 - move[0]:950 - move[0], 150 - move[1]:950 - move[1]]   # content moved by `move`
    r = track(img1, img2, patch=128, step=64, coarse=4)
    assert len(r["row"]) > 5
    assert np.median(r["d_row"]) == pytest.approx(move[0], abs=0.3)
    assert np.median(r["d_col"]) == pytest.approx(move[1], abs=0.3)


def test_unrelated_images_give_no_vectors():
    r = track(textured((512, 512), 1), textured((512, 512), 2))
    assert len(r["row"]) == 0


def test_patches_with_no_data_are_skipped():
    img = textured((512, 512))
    img[:, :] = np.nan
    assert len(track(img, img)["row"]) == 0


def arrows(rows, cols, d_rows, d_cols):
    t = dict(row=rows, col=cols, d_row=d_rows, d_col=d_cols, similarity=np.ones(len(rows)))
    t = {k: np.asarray(v, dtype="float64") for k, v in t.items()}
    t["source"] = "test"                                             # non-array entries are passed through
    return t


def test_keep_on_ice_uses_start_in_image1_and_end_in_image2():
    ice1, ice2 = np.ones((100, 100), bool), np.ones((100, 100), bool)
    ice1[10, 10] = False                                             # arrow 0 starts in water
    ice2[52, 47] = False                                             # arrow 1 ends in water (50+2.4, 50-3.4 -> 52, 47)
    t = arrows([10, 50, 70, 90], [10, 50, 70, 90], [0, 2.4, 1, 20], [0, -3.4, 1, 0])  # arrow 3 ends off the grid
    kept, keep = keep_on_ice(t, ice1, ice2)
    assert keep.tolist() == [False, False, True, False]
    assert kept["row"].tolist() == [70] and kept["source"] == "test"


def test_keep_on_ice_ignores_water_elsewhere():
    ice1 = np.ones((50, 50), bool)
    ice2 = ice1.copy()
    ice2[10, 10] = False                                             # water where the arrow STARTS, but in image 2
    kept, keep = keep_on_ice(arrows([10], [10], [5], [5]), ice1, ice2)
    assert keep.tolist() == [True]
    with pytest.raises(ValueError):
        keep_on_ice(arrows([1], [1], [0], [0]), ice1, ice2[:10])


def test_different_grids_fail_fast():
    with pytest.raises(ValueError):
        track(np.zeros((10, 10)), np.zeros((10, 12)))


def test_similarity_and_velocity():
    a = textured((32, 32))
    assert similarity(a, a) == pytest.approx(1.0)
    vx, vy, speed = to_velocity(np.array([-10.0]), np.array([0.0]), spacing_m=80, hours=24)
    assert vx[0] == 0 and vy[0] == pytest.approx(800 / 86400) and speed[0] == pytest.approx(0.8)
