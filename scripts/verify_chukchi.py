"""Check NISAR ice / open-water maps against an independent reference (AMSR2 ice concentration).

Run:  python scripts/verify_chukchi.py        (after download_scenes.py; add --no-db to skip MongoDB)
Prints one score line per NISAR scene and writes figures/chukchi_verification.png.

How the check works:
  1. Our map:   open water where HH is more than 3 dB below VV, else ice (scripts/classify.py).
  2. Reference: Uni Bremen AMSR2 ice concentration of the same day, 3.125 km cells.
  3. Score:     where the reference is clearly water (< 15 %) or clearly ice (>= 80 %), do we agree?
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mongo_store, reference                                                       # noqa: E402
from scripts.classify import COPOL_RATIO_WATER_DB, lee_filter, to_db, water_from_copol_ratio      # noqa: E402
from scripts.download_scenes import EDGE_SCENES, SCENES                                          # noqa: E402
from scripts.nisar_loader import load_gcov                                                       # noqa: E402

DATA, FIGS, REF = ROOT / "data", ROOT / "figures", ROOT / "data" / "reference"
NISAR_SCENES = EDGE_SCENES + [s for s in SCENES if s.startswith("NISAR")]
FIGURE_SCENE = EDGE_SCENES[0]
METHOD = f"HH - VV < {COPOL_RATIO_WATER_DB:.0f} dB = open water (after Lee 7x7)"
REGION_KM = (-2700, -300, -400, 1800)  # x0, x1, y0, y1 of the overview map, EPSG:3413
PLACES = {"Alaska": (-153, 67.5), "Chukotka": (176, 66.5), "Wrangel I.": (-179.5, 71.2),
          "Bering Strait": (-169, 65.8), "Chukchi Sea": (-166, 70.3), "Beaufort Sea": (-145, 73)}
WATER_C, LAND_C, ICE_C, MIXED_C = "#1f4e79", "#8c7b6b", "#ffffff", "#9ecae1"


def scene_date(name: str) -> str:
    return re.search(r"(\d{8})T\d{6}", name).group(1)


def analyse(name: str) -> dict:
    """Water mask, sampled reference and score for one NISAR scene."""
    s = load_gcov(DATA / f"{name}.h5")
    valid = np.isfinite(s.power["hh"]) & np.isfinite(s.power["vv"])
    db = {p: to_db(lee_filter(np.where(valid, s.power[p], np.nanmedian(s.power[p])), enl=s.looks[p])) for p in ("hh", "vv")}
    water = water_from_copol_ratio(db["hh"], db["vv"]) & valid
    conc, transform, crs = reference.read_bremen(reference.download(scene_date(name), "bremen", REF))
    ref = reference.sample(conc, transform, crs, s.x, s.y, s.epsg)
    score = reference.score_water_mask(water, ref, valid)
    sea = valid & (ref >= 0)
    score.update(our_water_pct=round(100 * float(water[valid].mean()), 1),
                 our_ice_pct=round(100 * float(1 - water[valid].mean()), 1),
                 reference_water_pct=round(100 * float(np.mean(ref[sea] < 15)), 1),
                 reference_mean_ice_concentration=round(float(ref[sea].mean()), 1))
    return dict(name=name, scene=s, valid=valid, db=db, water=water, ref=ref, score=score, reference_map=(conc, transform))


def footprint_km(valid, x, y):
    from scipy.spatial import ConvexHull
    rows, cols = np.where(valid)
    pts = np.c_[x[cols[::997]], y[rows[::997]]] / 1e3
    hull = pts[ConvexHull(pts).vertices]
    return np.r_[hull, hull[:1]]


def make_figure(results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch
    from pyproj import Transformer

    main = next(r for r in results if r["name"] == FIGURE_SCENE)
    s, valid = main["scene"], main["valid"]
    fig, axes = plt.subplots(1, 4, figsize=(27, 7.5))

    # 1. overview: reference map of the region with the scene outlines
    conc, transform = main["reference_map"]
    x0, x1, y0, y1 = (v * 1e3 for v in REGION_KM)
    c0, r0 = (int(v) for v in (~transform) * (x0, y1))
    c1, r1 = (int(v) for v in (~transform) * (x1, y0))
    region = conc[r0:r1, c0:c1]
    ax = axes[0]
    ax.set_facecolor(LAND_C)
    im = ax.imshow(np.ma.masked_where(region == reference.LAND, region), vmin=0, vmax=100, extent=REGION_KM,
                   cmap=LinearSegmentedColormap.from_list("ice", [WATER_C, MIXED_C, ICE_C]))
    plt.colorbar(im, ax=ax, label="ice concentration (%)", shrink=0.7)
    for r, colour, label in ((main, "red", "ice-edge scene (panels 2-4)"),
                             (next(q for q in results if q["name"] == SCENES[0]), "orange", "pack-ice scene (tracking)")):
        ax.plot(*footprint_km(r["valid"], r["scene"].x, r["scene"].y).T, color=colour, lw=2.5, label=label)
    to_map = Transformer.from_crs(4326, 3413, always_xy=True)
    for place, (lon, lat) in PLACES.items():
        px, py = to_map.transform(lon, lat)
        ax.text(px / 1e3, py / 1e3, place, ha="center", va="center", fontsize=10, bbox=dict(fc="white", alpha=0.75, ec="none", pad=1.5))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("1. Where are we? Reference map, 27 Oct 2026\nblue = open water, white = ice, brown = land")

    extent = [s.x[0] / 1e3, s.x[-1] / 1e3, s.y[-1] / 1e3, s.y[0] / 1e3]
    # 2. radar image
    axes[1].imshow(np.where(valid, main["db"]["hh"], np.nan)[::2, ::2], cmap="gray", vmin=-26, vmax=-11, extent=extent)
    axes[1].set_title("2. What NISAR sees (L-band HH, 27 Oct 14:18 UTC)\ndark = smooth (water), bright = rough (ice)")
    # 3. our map
    ours = np.where(valid, main["water"].astype(float), np.nan)
    axes[2].imshow(ours[::2, ::2], cmap=ListedColormap([ICE_C, WATER_C]), vmin=0, vmax=1, extent=extent, interpolation="nearest")
    sc = main["score"]
    axes[2].legend(handles=[Patch(fc=ICE_C, ec="k", label=f"sea ice ({sc['our_ice_pct']:.0f} %)"),
                            Patch(fc=WATER_C, ec="k", label=f"open water ({sc['our_water_pct']:.0f} %)"),
                            Patch(fc=LAND_C, ec="k", label="land (none in this scene)")], loc="lower right", fontsize=10)
    axes[2].set_title(f"3. Our map from NISAR (80 m pixels)\nrule: {METHOD.split(' (')[0]}")
    # 4. reference, same colours
    ref = np.where(valid, main["ref"], np.nan)
    classes = np.select([ref == reference.LAND, ref < 15, ref < 80, ref >= 80], [0, 1, 2, 3], np.nan)
    axes[3].imshow(classes[::2, ::2], cmap=ListedColormap([LAND_C, WATER_C, MIXED_C, ICE_C]), vmin=0, vmax=3, extent=extent, interpolation="nearest")
    axes[3].legend(handles=[Patch(fc=ICE_C, ec="k", label="ice (80 % or more)"), Patch(fc=MIXED_C, ec="k", label="mixed zone (15-80 %), not scored"),
                            Patch(fc=WATER_C, ec="k", label="open water (under 15 %)")], loc="lower right", fontsize=10)
    axes[3].set_title(f"4. Reference: AMSR2 ice concentration (3 km cells)\nagreement: water {sc['water_found_pct']} %, ice {sc['ice_found_pct']} %")
    for ax in axes:
        ax.set_xlabel("km"); ax.set_ylabel("km")
    for ax in axes[1:]:
        ax.set_facecolor("#dddddd")
    plt.tight_layout()
    FIGS.mkdir(exist_ok=True)
    plt.savefig(FIGS / "chukchi_verification.png", dpi=60)


def store(db, results):
    for r in results:
        doc = {"scene": r["name"], "method": METHOD, "channels": ["hh", "vv"], "threshold_db": COPOL_RATIO_WATER_DB,
               "water_pct": r["score"]["our_water_pct"], "ice_pct": r["score"]["our_ice_pct"],
               "reference": f"Uni Bremen AMSR2 ASI 3.125 km, {scene_date(r['name'])}", "reference_score": r["score"],
               "note": "Threshold tuned on the 27 Oct ice-edge scene; the other scenes are independent tests."}
        db.results.replace_one({"scene": r["name"], "method": METHOD}, doc, upsert=True)
    return len(results)


def main(use_db: bool):
    results = []
    for name in NISAR_SCENES:
        r = analyse(name)
        results.append(r)
        sc = r["score"]
        print(f"{re.search(r'[0-9]{8}T[0-9]{4}', name).group()}  our water {sc['our_water_pct']:5.1f} %  reference water {sc['reference_water_pct']:5.1f} %  "
              f"| water found {sc['water_found_pct']} %  ice found {sc['ice_found_pct']} %  balanced {sc['balanced_accuracy_pct']} %")
    make_figure(results)
    if use_db:
        print("MongoDB results stored:", store(mongo_store.get_db(), results))


if __name__ == "__main__":
    main(use_db="--no-db" not in sys.argv)
