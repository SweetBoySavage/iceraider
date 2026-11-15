import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ice_tracking import similarity, to_velocity, track


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


def test_different_grids_fail_fast():
    with pytest.raises(ValueError):
        track(np.zeros((10, 10)), np.zeros((10, 12)))


def test_similarity_and_velocity():
    a = textured((32, 32))
    assert similarity(a, a) == pytest.approx(1.0)
    vx, vy, speed = to_velocity(np.array([-10.0]), np.array([0.0]), spacing_m=80, hours=24)
    assert vx[0] == 0 and vy[0] == pytest.approx(800 / 86400) and speed[0] == pytest.approx(0.8)
