"""Ice / open-water detection: features, labelled tables and the fixed train/test protocol.

Build the training table:   python scripts/detection.py table train
(The test table is built only once, after the model is frozen:  python scripts/detection.py table test)

Label of a pixel (see scripts/reference.py):
  OPEN WATER: AMSR2 cell and all its neighbours below 15 % ice.
  ICE:        AMSR2 cell and all its neighbours at 80 % or more.
  ICE, confirmed twice: as above AND an ICESat-2 laser segment in that pixel says ice (clear sky, good fit).
Everything else is 'unsure' and is not used.
"""
from __future__ import annotations

import json
import re
import sys
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reference                         # noqa: E402
from scripts.classify import lee_filter, to_db        # noqa: E402
from scripts.nisar_loader import load_gcov            # noqa: E402

DATA, REF, TABLES = ROOT / "data", ROOT / "data" / "reference", ROOT / "data" / "tables"
FEATURES = ["hh", "vv", "ratio", "incidence", "hh_texture", "ratio_texture", "hh_smooth", "ratio_smooth"]
PER_CLASS = 20000          # random pixels per class and scene
TEXTURE_WINDOW = 9         # pixels (720 m)
ICESAT2_MAX_HOURS = 6


def local_std(img: np.ndarray, size: int) -> np.ndarray:
    mean = uniform_filter(img, size)
    return np.sqrt(np.maximum(uniform_filter(img * img, size) - mean * mean, 0))


def features(scene) -> tuple[dict, np.ndarray]:
    """Feature images (dB or degrees) and the mask of pixels where all of them are usable."""
    valid = np.isfinite(scene.power["hh"]) & np.isfinite(scene.power["vv"])
    db = {p: to_db(lee_filter(np.where(valid, scene.power[p], np.nanmedian(scene.power[p])), enl=scene.looks[p]))
          for p in ("hh", "vv")}
    ratio = db["hh"] - db["vv"]
    out = {"hh": db["hh"], "vv": db["vv"], "ratio": ratio, "incidence": scene.incidence,
           "hh_texture": local_std(db["hh"], TEXTURE_WINDOW), "ratio_texture": local_std(ratio, TEXTURE_WINDOW),
           "hh_smooth": uniform_filter(db["hh"], TEXTURE_WINDOW), "ratio_smooth": uniform_filter(ratio, TEXTURE_WINDOW)}
    # windows that reach outside the image mix in fill values: drop a border of half a window
    inner = uniform_filter(valid.astype("float32"), TEXTURE_WINDOW + 6) > 0.999
    return {k: v.astype("float32") for k, v in out.items()}, valid & inner


def scene_time(name: str) -> datetime:
    return datetime.strptime(re.search(r"\d{8}T\d{6}", name).group(), "%Y%m%dT%H%M%S")


def find_icesat2(name: str, scene) -> list[str]:
    """Download ICESat-2 quick-look files that cross the scene within ICESAT2_MAX_HOURS. Returns local paths."""
    import requests
    from pyproj import Transformer
    t = scene_time(name)
    lon, lat = Transformer.from_crs(scene.epsg, 4326, always_xy=True).transform(
        [scene.x[0], scene.x[-1], scene.x[-1], scene.x[0]], [scene.y[0], scene.y[0], scene.y[-1], scene.y[-1]])
    window = timedelta(hours=ICESAT2_MAX_HOURS)
    entries = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", timeout=120, params=dict(
        short_name="ATL07QL", page_size=50, bounding_box=f"{min(lon)},{min(lat)},{max(lon)},{max(lat)}",
        temporal=f"{t - window:%Y-%m-%dT%H:%M:%SZ},{t + window:%Y-%m-%dT%H:%M:%SZ}")).json()["feed"]["entry"]
    paths = []
    for e in entries:
        url = next(l["href"] for l in e["links"] if l["href"].startswith("https") and l["href"].endswith(".h5"))
        path = REF / "icesat2" / url.rsplit("/", 1)[1]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.Session().get(url, timeout=900)
            r.raise_for_status()
            path.write_bytes(r.content)
        paths.append(str(path))
    return paths


def scene_labels(name: str, scene) -> tuple[np.ndarray, np.ndarray]:
    """(label, confirmed_by_icesat2) on the scene grid."""
    date = scene_time(name).strftime("%Y%m%d")
    conc, transform, crs = reference.read_bremen(reference.download(date, "bremen", REF))
    cells = reference.certain_labels(conc).astype("float32")
    label = reference.sample(cells, transform, crs, scene.x, scene.y, scene.epsg)
    label = np.where(np.isfinite(label), label, reference.UNSURE).astype("int8")
    laser = np.zeros(label.shape, bool)
    for path in find_icesat2(name, scene):
        laser |= reference.points_to_mask(*reference.read_icesat2_ice(path), scene.x, scene.y, scene.epsg)
    return label, laser & (label == reference.ICE)


def scene_rows(name: str, per_class: int = PER_CLASS):
    """Sampled feature rows of one scene: X, y, confirmed-by-laser flag."""
    scene = load_gcov(DATA / f"{name}.h5")
    feats, usable = features(scene)
    label, laser = scene_labels(name, scene)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    picks = []
    for cls in (reference.OPEN_WATER, reference.ICE):
        idx = np.flatnonzero(usable & (label == cls) & ~laser)
        picks.append(rng.choice(idx, min(per_class, len(idx)), replace=False))
    picks.append(np.flatnonzero(usable & laser))                    # keep every laser-confirmed pixel
    idx = np.concatenate(picks)
    X = np.column_stack([feats[k].ravel()[idx] for k in FEATURES])
    return X, label.ravel()[idx], laser.ravel()[idx]


def build_table(split: str) -> Path:
    scenes = json.loads((DATA / "dataset.json").read_text())[split]
    X, y, laser, scene_id = [], [], [], []
    for i, s in enumerate(scenes):
        xs, ys, ls = scene_rows(s["name"])
        X.append(xs); y.append(ys); laser.append(ls); scene_id.append(np.full(len(ys), i))
        print(f"{split} {i + 1:2d}/{len(scenes)} {s['date']} {s['type']:9s} water {np.sum(ys == 0):6d}  ice {np.sum(ys == 1):6d}  "
              f"laser-confirmed ice {ls.sum():5d}", flush=True)
    TABLES.mkdir(exist_ok=True)
    out = TABLES / f"{split}.npz"
    np.savez_compressed(out, X=np.concatenate(X), y=np.concatenate(y), laser=np.concatenate(laser),
                        scene=np.concatenate(scene_id), features=FEATURES, names=[s["name"] for s in scenes],
                        types=[s["type"] for s in scenes])
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "table" and sys.argv[2] in ("train", "test"):
        print("saved", build_table(sys.argv[2]))
    else:
        sys.exit(__doc__)
