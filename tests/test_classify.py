import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.classify import dark_threshold, kmeans_classes, largest_clean_square, lee_filter, to_db


def test_to_db_and_floor():
    assert to_db(np.array([1.0, 0.01, 0.0, -1.0])).tolist() == [0.0, -20.0, -40.0, -40.0]


def test_lee_filter_smooths_speckle_but_keeps_the_mean():
    rng = np.random.default_rng(0)
    img = rng.gamma(shape=10, scale=0.1 / 10, size=(200, 200))     # uniform area, 10 looks, mean 0.1
    out = lee_filter(img, enl=10)
    assert out.mean() == pytest.approx(0.1, rel=0.02) and out.std() < img.std() / 3


def test_kmeans_orders_classes_dark_to_bright_and_threshold_is_between_them():
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(-20, 0.3, 5000), rng.normal(-12, 0.3, 5000)]).reshape(100, 100)
    labels, centres = kmeans_classes(a, a, k=2)
    assert labels[0, 0] == 0 and labels[-1, -1] == 1 and centres[0, 0] < centres[1, 0]
    assert dark_threshold(centres) == pytest.approx(-16, abs=0.3)


def test_largest_clean_square_avoids_no_data():
    valid = np.ones((300, 300), bool)
    valid[:, :100] = False
    assert largest_clean_square(valid, sizes=(250, 150), step=50) == (0, 100, 150)
    with pytest.raises(ValueError):
        largest_clean_square(~np.ones((300, 300), bool), sizes=(150,))
