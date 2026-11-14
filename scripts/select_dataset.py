"""Choose the NISAR scenes for training and testing the ice / open-water model. Writes config/dataset.json.

Rules (fixed before any test scene is opened):
  - Area: Chukchi + western Beaufort Sea, 70-78 N, 150-180 W. Only scenes with HH and VV, no land.
  - TRAIN: 14 Sep - 28 Oct 2026.   TEST: 29 - 31 Oct 2026 (strictly later; never used for tuning).
  - Scene type comes from the AMSR2 reference of the same day: edge (water and ice), all ice, all water.
  - Scenes with an ICESat-2 pass within 6 h, and scenes already on disk, are taken first.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reference                      # noqa: E402

AREA_WKT = "POLYGON((-180 70,-150 70,-150 78,-180 78,-180 70))"
TRAIN = ("2026-09-14", "2026-10-28")
TEST = ("2026-10-29", "2026-10-31")
WANT = {"train": {"edge": 14, "all ice": 5, "all water": 5}, "test": {"edge": 8, "all ice": 2, "all water": 2}}
# NISAR scenes with an ICESat-2 (ATL07 quick look) pass within 6 hours, found with NASA's CMR search.
ICESAT2_MATCHES = ["033_131_D_051", "033_131_D_052", "034_074_D_049", "034_075_D_046",
                   "034_102_D_050", "034_102_D_051", "034_103_D_047", "034_104_A_044"]
OUT = ROOT / "config" / "dataset.json"


def scene_type(footprint_lonlat, date: str, cache: dict):
    """'edge' (over 10 % clear water and over 10 % clear ice), 'all ice', 'all water', 'mixed', or None (land)."""
    from pyproj import Transformer
    from shapely import contains_xy
    from shapely.geometry import Polygon
    if date not in cache:
        cache[date] = reference.read_bremen(reference.download(date, "bremen", ROOT / "data" / "reference"))
    conc, transform, crs = cache[date]
    c = np.array(footprint_lonlat)
    x, y = Transformer.from_crs(4326, 3413, always_xy=True).transform(c[:, 0], c[:, 1])
    poly = Polygon(np.c_[x, y]).buffer(0)
    x0, y0, x1, y1 = poly.bounds
    xs, ys = np.arange(x0, x1, 6250), np.arange(y1, y0, -6250)
    values = reference.sample(conc, transform, crs, xs, ys, 3413)[contains_xy(poly, *np.meshgrid(xs, ys))]
    if len(values) < 50 or np.mean(values == reference.LAND) > 0.005:
        return None
    water, ice = np.mean(values < 15), np.mean(values >= 80)
    if water > 0.10 and ice > 0.10:
        return "edge"
    return "all ice" if ice > 0.95 else "all water" if water > 0.95 else "mixed"


def choose(candidates, want, on_disk, rng):
    """Pick `want[type]` scenes per type: ICESat-2 matches and files on disk first, then spread over the dates."""
    chosen = [c for c in candidates if c["icesat2"]]             # always keep scenes with a laser pass
    for kind, n in want.items():
        n -= sum(c["type"] == kind for c in chosen)
        pool = [c for c in candidates if c["type"] == kind and c not in chosen]
        first = [c for c in pool if c["name"] in on_disk]
        rest = [c for c in pool if c not in first]
        rng.shuffle(rest)
        rest.sort(key=lambda c: sum(c["date"] == d["date"] for d in first))      # prefer new dates
        seen_dates, spread, later = {c["date"] for c in first}, [], []
        for c in rest:
            (later if c["date"] in seen_dates else spread).append(c)
            seen_dates.add(c["date"])
        chosen += (first + spread + later)[:max(n, 0)]
    return chosen


def main():
    import asf_search as asf
    cache, sets = {}, {}
    on_disk = {p.stem for p in (ROOT / "data").glob("NISAR*.h5")}
    for split, (start, end) in (("train", TRAIN), ("test", TEST)):
        found = {r.properties["sceneName"]: r for r in asf.search(
            platform="NISAR", processingLevel="GCOV", start=start, end=end + "T23:59:59Z", intersectsWith=AREA_WKT, maxResults=10000)}
        candidates = []
        for name, r in sorted(found.items()):
            mode = name.split("_")[9]
            if "_PR_" not in name or {mode[:2], mode[2:]} != {"SH", "SV"}:
                continue
            date = name.split("_")[11][:8]
            if not start.replace("-", "") <= date <= end.replace("-", ""):
                continue
            kind = scene_type(r.geometry["coordinates"][0], date, cache)
            if kind:
                candidates.append({"name": name, "date": date, "type": kind, "icesat2": name[17:30] in ICESAT2_MATCHES,
                                   "gb": round(r.properties["bytes"][r.properties["fileName"]]["bytes"] / 1e9, 2)})
        sets[split] = choose(candidates, WANT[split], on_disk, random.Random(2026))
        print(f"{split}: {len(candidates)} candidates -> {len(sets[split])} chosen, "
              f"{sum(c['gb'] for c in sets[split]):.1f} GB, dates {sorted({c['date'][4:] for c in sets[split]})}")
    assert max(c["date"] for c in sets["train"]) < min(c["date"] for c in sets["test"]), "train must end before test starts"
    OUT.write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "train_period": TRAIN,
                               "test_period": TEST, **sets}, indent=1))


if __name__ == "__main__":
    main()
