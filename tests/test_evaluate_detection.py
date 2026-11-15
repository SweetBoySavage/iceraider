import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_detection as ev
from scripts.detection import FEATURES
from scripts.train_detection import ICE, WATER


class RecordingModel:
    """Calls every pixel open water and remembers what it was shown."""

    def __init__(self):
        self.seen = []

    def predict(self, X):
        self.seen.append(X.copy())
        return np.full(len(X), WATER)


def fake_scene(n=80):
    rng = np.random.default_rng(0)
    hh, vv = rng.gamma(39, 0.02 / 39, (n, n)), rng.gamma(39, 0.06 / 39, (n, n))
    hh[:, :10] = np.nan                                               # a strip without data
    return SimpleNamespace(power={"hh": hh, "vv": vv}, looks={"hh": 39.0, "vv": 39.0}, incidence=np.full((n, n), 40.0),
                           x=np.arange(n) * 80.0, y=np.arange(n, 0, -1) * 80.0, epsg=3413)


def test_land_is_painted_over_the_prediction_and_never_changes_the_model_input(monkeypatch):
    scene, model = fake_scene(), RecordingModel()
    frozen = {"model": model, "features": FEATURES}
    land = np.zeros((80, 80), bool)
    land[:, 50:] = True
    land[:, :5] = True                                                # land where there is no data stays no data
    asked = []
    monkeypatch.setattr(ev.land_mask, "land_mask", lambda x, y, epsg, margin_m=None: asked.append(margin_m) or land)

    classes = ev.classify_scene(frozen, scene, margin_m=2500)
    assert asked == [2500]
    assert (classes[:, 50:] == ev.LAND).all()                         # the model said water there; land wins
    assert (classes[:, :10] == ev.NO_DATA).all()
    assert (classes[40, 20:50] == ev.OPEN_WATER).all()

    monkeypatch.setattr(ev.land_mask, "land_mask", lambda x, y, epsg, margin_m=None: np.zeros((80, 80), bool))
    without_land = ev.classify_scene(frozen, scene)
    assert not (without_land == ev.LAND).any()
    assert np.array_equal(model.seen[0], model.seen[1])               # same model input with and without land


def test_classes_keep_ice_and_water_apart(monkeypatch):
    scene = fake_scene()
    model = SimpleNamespace(predict=lambda X: np.full(len(X), ICE))
    monkeypatch.setattr(ev.land_mask, "land_mask", lambda x, y, epsg, margin_m=None: np.zeros((80, 80), bool))
    classes = ev.classify_scene({"model": model, "features": FEATURES}, scene)
    assert set(np.unique(classes)) == {ev.SEA_ICE, ev.NO_DATA}
