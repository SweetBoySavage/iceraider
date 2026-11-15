import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.land_mask import HEADER, add_margin, land_mask, rasterize_land, read_gshhg


def polygon_bytes(points, level, north):
    head = HEADER.pack(0, len(points), level, 0, 0, 0, int(north * 1e6), 0, 0, -1, -1)
    return head + b"".join(struct.pack(">2i", int(lon * 1e6), int(lat * 1e6)) for lon, lat in points)


def square(lon0, lat0, size=1.0):
    return [(lon0, lat0), (lon0 + size, lat0), (lon0 + size, lat0 + size), (lon0, lat0 + size), (lon0, lat0)]


def test_reads_only_the_wanted_level_and_latitude(tmp_path):
    f = tmp_path / "g.b"
    f.write_bytes(polygon_bytes(square(200, 70), 1, 71) + polygon_bytes(square(10, 10), 1, 11) + polygon_bytes(square(201, 70.2, 0.1), 2, 70.3))
    polys = read_gshhg(f, min_north=55)
    assert len(polys) == 1 and polys[0][0].tolist() == [200.0, 70.0] and len(polys[0]) == 5
    assert len(read_gshhg(f)) == 2
    f.write_bytes(f.read_bytes()[:-4])
    with pytest.raises(ValueError):
        read_gshhg(f)


def test_rasterize_and_margin():
    x, y = np.arange(0, 2000, 100.0), np.arange(2000, 0, -100.0)
    land = rasterize_land([np.array([[500, 500], [1000, 500], [1000, 1000], [500, 1000], [500, 500.0]])], x, y)
    assert land[np.argmin(abs(y - 700)), np.argmin(abs(x - 700))] and not land[0, 0]
    wide = add_margin(land, 100.0, margin_m=300)
    row = np.argmin(abs(y - 700))
    assert wide[row, np.argmin(abs(x - 300))] and not wide[row, np.argmin(abs(x - 100))]
    assert (add_margin(land, 100.0, margin_m=0) == land).all() and not rasterize_land([], x, y).any()


def test_land_mask_places_an_island_correctly_across_the_180_degree_line(tmp_path):
    from pyproj import Transformer
    f = tmp_path / "g.b"
    f.write_bytes(polygon_bytes(square(179.5, 71.0), 1, 72))          # island from 179.5 E to 179.5 W
    to_map = Transformer.from_crs(4326, 3413, always_xy=True)
    cx, cy = to_map.transform(-179.9, 71.5)                           # inside, on the western side of the line
    x, y = np.arange(cx - 40_000, cx + 40_000, 400.0), np.arange(cy + 40_000, cy - 40_000, -400.0)
    mask = land_mask(x, y, 3413, margin_m=0, path=f)
    assert mask[len(y) // 2, len(x) // 2]
    ox, oy = to_map.transform(-178.5, 71.5)                           # 1 degree east of the island: sea
    assert not land_mask(np.arange(ox - 2000, ox + 2000, 400.0), np.arange(oy + 2000, oy - 2000, -400.0), 3413, margin_m=0, path=f).any()


def test_missing_coast_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        land_mask(np.arange(3.0), np.arange(3.0, 0, -1), 3413, path=tmp_path / "none.b")


def test_margin_comes_from_environment_then_env_file_then_default(tmp_path, monkeypatch):
    from scripts.land_mask import MARGIN_ENV, MARGIN_M, margin_from_env
    monkeypatch.delenv(MARGIN_ENV, raising=False)
    env = tmp_path / ".env"
    assert margin_from_env(env) == MARGIN_M                       # no file, no variable
    env.write_text("MONGODB_URI=secret\nLAND_MARGIN_M=2500   # metres\n")
    assert margin_from_env(env) == 2500.0
    monkeypatch.setenv(MARGIN_ENV, "5000")
    assert margin_from_env(env) == 5000.0                         # the environment variable wins
    monkeypatch.setenv(MARGIN_ENV, "")
    assert margin_from_env(tmp_path / "none") == MARGIN_M         # empty counts as not set


@pytest.mark.parametrize("bad", ["-1", "wide", "nan", "inf"])
def test_bad_margin_fails_fast(bad, tmp_path, monkeypatch):
    from scripts.land_mask import MARGIN_ENV, margin_from_env
    monkeypatch.setenv(MARGIN_ENV, bad)
    with pytest.raises(ValueError, match=MARGIN_ENV):
        margin_from_env(tmp_path / "none")


def test_small_margin_warns(tmp_path, monkeypatch):
    from scripts.land_mask import MARGIN_ENV, margin_from_env
    monkeypatch.setenv(MARGIN_ENV, "100")
    with pytest.warns(UserWarning, match="contain land"):
        assert margin_from_env(tmp_path / "none") == 100.0


def test_land_mask_uses_the_margin_from_the_environment(tmp_path, monkeypatch):
    from pyproj import Transformer
    from scripts.land_mask import MARGIN_ENV
    f = tmp_path / "g.b"
    f.write_bytes(polygon_bytes(square(-160.0, 71.0, 0.2), 1, 72))
    cx, cy = Transformer.from_crs(4326, 3413, always_xy=True).transform(-159.9, 71.1)
    x, y = np.arange(cx - 30_000, cx + 30_000, 400.0), np.arange(cy + 30_000, cy - 30_000, -400.0)
    monkeypatch.setenv(MARGIN_ENV, "1000")
    narrow = land_mask(x, y, 3413, path=f)
    monkeypatch.setenv(MARGIN_ENV, "5000")
    wide = land_mask(x, y, 3413, path=f)
    assert (narrow == land_mask(x, y, 3413, margin_m=1000, path=f)).all()
    assert wide.sum() > narrow.sum() and wide[narrow].all()       # a wider margin only adds land


def test_land_and_margin_are_separate_and_land_outside_the_grid_still_counts(tmp_path):
    from pyproj import Transformer
    from scripts.land_mask import land_and_margin
    f = tmp_path / "g.b"
    f.write_bytes(polygon_bytes(square(-160.0, 71.0, 0.2), 1, 72))
    cx, cy = Transformer.from_crs(4326, 3413, always_xy=True).transform(-159.9, 71.1)
    x, y = np.arange(cx - 30_000, cx + 30_000, 400.0), np.arange(cy + 30_000, cy - 30_000, -400.0)
    land, margin = land_and_margin(x, y, 3413, margin_m=2000, path=f)
    assert land.any() and margin.any() and not (land & margin).any()
    assert (land == land_mask(x, y, 3413, margin_m=0, path=f)).all()
    assert ((land | margin) == land_mask(x, y, 3413, margin_m=2000, path=f)).all()
    # a grid that starts just past the island: no land inside it, but the margin reaches in
    rows, cols = np.nonzero(land)
    start = cols.max() + 2
    land2, margin2 = land_and_margin(x[start:], y, 3413, margin_m=2000, path=f)
    assert not land2.any() and margin2.any()
    assert (margin2 == margin[:, start:]).all()
