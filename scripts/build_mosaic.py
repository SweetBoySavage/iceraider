"""Stitch the ice / open-water maps of several NISAR scenes into one map for the route interface.

    python scripts/build_mosaic.py                      # 27 - 31 Oct 2026, the period used by the interface
    python scripts/build_mosaic.py 20261027 20261031    # any other period (first day, last day)

Writes data/maps/mosaic_<first>_<last>.npz and figures/mosaic_<first>_<last>.png. Read it back with `load_mosaic`.

How the map is made:
- Every downloaded scene of the period is classified at 80 m by the frozen model, exactly as in the test. Scenes with a dead channel are skipped.
- The answers are collected in cells of CELL_M. A cell is sea ice when at least MIN_ICE_FRACTION of it is ice:
  for a ship a cell that is partly ice is not open water.
- Where scenes overlap, the newest one wins. `scene_index` tells which scene every cell came from.
- Land and the coast margin come from the coastline, once, for the whole map (also outside the scenes).
- `ice_distance_m` is the distance from every cell to the nearest sea ice cell, so a safety margin around
  the ice of any width is one comparison in the interface.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detection, land_mask                                              # noqa: E402
from scripts.evaluate_detection import (COAST_MARGIN, FIGS, LAND, MAPS, NO_DATA, OPEN_WATER, SEA_ICE,   # noqa: E402
                                        draw_classes, load_model, predict_scene)
from scripts.nisar_loader import DeadChannelError, load_gcov                          # noqa: E402

CELL_M = 400.0               # size of a mosaic cell
MIN_ICE_FRACTION = 0.25      # a cell is sea ice from this share of ice pixels
MIN_SEEN_FRACTION = 0.5      # a cell with fewer usable pixels than this share stays "no data"
PAD_M = 100_000.0            # empty border around the scenes, so the coast and the route ends are in view
MAX_DISTANCE_M = 65_000      # distances are stored as uint16 metres and capped here
NO_SCENE = 255
EPSG = 3413
FIRST_DAY, LAST_DAY = "20261027", "20261031"


@dataclass(frozen=True)
class Grid:
    """North-up grid of square cells. (x0, y0) is the outer corner of the top-left cell, in metres."""
    x0: float
    y0: float
    nx: int
    ny: int
    cell_m: float = CELL_M

    @property
    def x(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell_m

    @property
    def y(self) -> np.ndarray:
        return self.y0 - (np.arange(self.ny) + 0.5) * self.cell_m


def grid_around(bounds: list[tuple[float, float, float, float]], cell_m: float = CELL_M, pad_m: float = PAD_M) -> Grid:
    """Smallest grid, snapped to whole cells, that holds every (xmin, xmax, ymin, ymax) plus the border."""
    if not bounds:
        raise ValueError("no scenes to build a grid around")
    b = np.asarray(bounds, float)
    x0 = np.floor((b[:, 0].min() - pad_m) / cell_m) * cell_m
    x1 = np.ceil((b[:, 1].max() + pad_m) / cell_m) * cell_m
    y0 = np.ceil((b[:, 3].max() + pad_m) / cell_m) * cell_m
    y1 = np.floor((b[:, 2].min() - pad_m) / cell_m) * cell_m
    return Grid(float(x0), float(y0), int(round((x1 - x0) / cell_m)), int(round((y0 - y1) / cell_m)), cell_m)


def bin_scene(ice: np.ndarray, usable: np.ndarray, x: np.ndarray, y: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Count per mosaic cell the ice pixels and the usable pixels of one scene (pixels outside the grid are dropped)."""
    col = np.floor((np.asarray(x) - grid.x0) / grid.cell_m).astype(int)
    row = np.floor((grid.y0 - np.asarray(y)) / grid.cell_m).astype(int)
    inside = ((row >= 0) & (row < grid.ny))[:, None] & ((col >= 0) & (col < grid.nx))[None, :]
    keep = usable & inside
    cell = (row[:, None] * grid.nx + col[None, :])[keep]
    size = grid.nx * grid.ny
    seen = np.bincount(cell, minlength=size).reshape(grid.ny, grid.nx)
    ice_n = np.bincount(cell, weights=ice[keep], minlength=size).reshape(grid.ny, grid.nx)
    return ice_n, seen


def cell_classes(ice_n: np.ndarray, seen: np.ndarray, pixels_per_cell: float,
                 min_ice: float = MIN_ICE_FRACTION, min_seen: float = MIN_SEEN_FRACTION) -> np.ndarray:
    """SEA_ICE / OPEN_WATER / NO_DATA per cell from the pixel counts of one scene."""
    out = np.full(seen.shape, NO_DATA, "uint8")
    known = seen >= min_seen * pixels_per_cell
    out[known] = np.where(ice_n[known] >= min_ice * seen[known], SEA_ICE, OPEN_WATER)
    return out


def paint(classes: np.ndarray, scene_index: np.ndarray, new: np.ndarray, index: int) -> None:
    """Put one scene on the mosaic, in place. Call in time order: a newer scene covers an older one."""
    known = new != NO_DATA
    classes[known] = new[known]
    scene_index[known] = index


def ice_distance_m(classes: np.ndarray, cell_m: float, cap_m: int = MAX_DISTANCE_M) -> np.ndarray:
    """Distance in metres from each cell to the nearest sea ice cell (0 on ice), capped at `cap_m`, as uint16."""
    ice = classes == SEA_ICE
    if not ice.any():
        return np.full(classes.shape, cap_m, "uint16")
    return np.minimum(distance_transform_edt(~ice) * cell_m, cap_m).astype("uint16")


def scenes_between(first_day: str, last_day: str, data_dir: Path = detection.DATA) -> list[str]:
    """Names of every downloaded NISAR scene from first_day to last_day (YYYYMMDD), oldest first.

    Takes all files in the data folder, not only the training and test list: a coastal scene is welcome on the map.
    """
    names = [p.stem for p in Path(data_dir).glob("NISAR_*.h5")
             if first_day <= f"{detection.scene_time(p.stem):%Y%m%d}" <= last_day]
    if not names:
        raise ValueError(f"no NISAR scenes between {first_day} and {last_day} in {data_dir}")
    return sorted(names, key=detection.scene_time)


def build(first_day: str = FIRST_DAY, last_day: str = LAST_DAY, margin_m: float | None = None) -> dict:
    frozen = load_model()
    margin_m = land_mask.margin_from_env() if margin_m is None else margin_m
    loaded, skipped = [], []
    for name in scenes_between(first_day, last_day):
        try:
            scene = load_gcov(detection.DATA / f"{name}.h5")
        except DeadChannelError as error:
            skipped.append(name)
            print("SKIPPED", error, flush=True)
            continue
        if scene.epsg != EPSG:
            raise ValueError(f"{name}: EPSG {scene.epsg}, expected {EPSG}")
        water, usable = predict_scene(frozen, scene)
        loaded.append((name, ~water & usable, usable, scene.x, scene.y, scene.spacing_m))
        print("classified", name, flush=True)

    grid = grid_around([(x.min(), x.max(), y.min(), y.max()) for _, _, _, x, y, _ in loaded])
    classes = np.full((grid.ny, grid.nx), NO_DATA, "uint8")
    scene_index = np.full((grid.ny, grid.nx), NO_SCENE, "uint8")
    for i, (name, ice, usable, x, y, spacing) in enumerate(loaded):
        ice_n, seen = bin_scene(ice, usable, x, y, grid)
        paint(classes, scene_index, cell_classes(ice_n, seen, (grid.cell_m / spacing) ** 2), i)

    distance = ice_distance_m(classes, grid.cell_m)          # from the radar answer only, before land is painted
    land, margin = land_mask.land_and_margin(grid.x, grid.y, EPSG, margin_m)
    classes[margin] = COAST_MARGIN
    classes[land] = LAND
    return {"classes": classes, "scene_index": scene_index, "ice_distance_m": distance, "x": grid.x, "y": grid.y,
            "epsg": EPSG, "cell_m": grid.cell_m, "margin_m": margin_m,
            "scenes": [name for name, *_ in loaded], "skipped": skipped}


def mosaic_path(first_day: str, last_day: str) -> Path:
    return MAPS / f"mosaic_{first_day}_{last_day}.npz"


def save_mosaic(mosaic: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **mosaic)
    return path


def load_mosaic(first_day: str = FIRST_DAY, last_day: str = LAST_DAY) -> dict:
    """The saved mosaic: classes, scene_index, ice_distance_m, x, y (cell centres, metres), scenes, skipped, ..."""
    path = mosaic_path(first_day, last_day)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/build_mosaic.py {first_day} {last_day}")
    with np.load(path) as f:
        out = {k: f[k] for k in ("classes", "scene_index", "ice_distance_m", "x", "y")}
        out |= {"epsg": int(f["epsg"]), "cell_m": float(f["cell_m"]), "margin_m": float(f["margin_m"]),
                "scenes": [str(s) for s in f["scenes"]], "skipped": [str(s) for s in f["skipped"]]}
    return out


def figure(mosaic: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, classes = mosaic["x"], mosaic["y"], mosaic["classes"]
    extent = [x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3]
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor("#dddddd")
    draw_classes(ax, classes, extent, step=2)
    for i, name in enumerate(mosaic["scenes"]):
        rows, cols = np.nonzero(mosaic["scene_index"] == i)
        if rows.size:
            ax.text(x[int(cols.mean())] / 1e3, y[int(rows.mean())] / 1e3, f"{detection.scene_time(name):%d %b\n%H:%M}",
                    ha="center", va="center", fontsize=8, color="#c0392b", weight="bold")
    ax.set_title(f"NISAR mosaic, {len(mosaic['scenes'])} scenes ({len(mosaic['skipped'])} skipped). White = sea ice, "
                 "blue = open water,\nbrown = land, yellow = coast margin, grey = no data. Red = image time (UTC)")
    ax.set_xlabel("km"); ax.set_ylabel("km")
    plt.tight_layout(); plt.savefig(out, dpi=80, bbox_inches="tight"); plt.close(fig)
    return out


def main(argv: list[str]) -> None:
    first_day, last_day = (argv + [FIRST_DAY, LAST_DAY])[:2] if len(argv) != 1 else (argv[0], argv[0])
    mosaic = build(first_day, last_day)
    print("saved", save_mosaic(mosaic, mosaic_path(first_day, last_day)))
    print("saved", figure(mosaic, FIGS / f"mosaic_{first_day}_{last_day}.png"))
    classes, km2 = mosaic["classes"], (mosaic["cell_m"] / 1e3) ** 2
    for cls, label in ((SEA_ICE, "sea ice"), (OPEN_WATER, "open water")):
        print(f"{label:11s} {np.sum(classes == cls) * km2:10,.0f} km2")


if __name__ == "__main__":
    main(sys.argv[1:])
