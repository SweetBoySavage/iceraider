import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.mongo_store import mongo_uri, motion_docs, result_doc, scene_doc


def test_uri_comes_from_env_or_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(RuntimeError):
        mongo_uri(tmp_path / "missing.env")
    f = tmp_path / ".env"
    f.write_text("OTHER=1\nMONGODB_URI='mongodb://x'\n")
    assert mongo_uri(f) == "mongodb://x"
    monkeypatch.setenv("MONGODB_URI", "mongodb://y")
    assert mongo_uri(f) == "mongodb://y"


def test_scene_doc_rejects_unknown_sensor():
    with pytest.raises(ValueError):
        scene_doc("a", "Landsat", "x", datetime(2026, 1, 1), "f", {}, [], "L", "EPSG:3413")


def test_result_doc_uses_percentages():
    d = result_doc("s", "kmeans", ["hh", "vv"], [0.1, 0.9], -17.64, 0.098, (1, 2, 3), "n")
    assert d["class_shares_pct"] == [10.0, 90.0] and d["dark_pct"] == 9.8 and d["bright_ice_pct"] == 90.2


def test_motion_docs_speed_direction_and_position():
    # EPSG:3413: at x=0, y<0 we are on 45 W; moving towards +y there means moving north (to the pole).
    x = np.array([0.0, 80.0])
    y = np.array([-1_500_000.0, -1_500_080.0])
    track = {"row": np.array([1.0]), "col": np.array([0.0]), "d_row": np.array([-125.0]), "d_col": np.array([0.0]),
             "similarity": np.array([0.8])}
    d = motion_docs(track, x, y, 3413, 80.0, datetime(2026, 10, 27), 12.0, "NISAR tracking", "a", "b")[0]
    assert d["location"]["coordinates"][0] == pytest.approx(-45.0, abs=1e-6)
    assert d["displacement_km"] == pytest.approx(10.0, rel=0.05)      # map scale differs a little from true distance
    assert d["speed_km_day"] == pytest.approx(20.0, rel=0.05)
    assert d["direction_deg"] == pytest.approx(0.0, abs=0.5) or d["direction_deg"] == pytest.approx(360.0, abs=0.5)
    assert d["date_end"] == datetime(2026, 10, 27, 12)
