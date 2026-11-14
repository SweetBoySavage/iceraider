"""Download the test scenes (NISAR GCOV + Sentinel-1 EW GRD) from NASA ASF DAAC into data/.

Needs a NASA Earthdata Login in ~/.netrc:
    machine urs.earthdata.nasa.gov login <user> password <pass>

Run:  python scripts/download_scenes.py            # downloads everything in SCENES
      python scripts/download_scenes.py --dry-run  # only lists names and sizes
      python scripts/download_scenes.py --dataset  # also the train/test scenes in config/dataset.json

Area: Chukchi Sea / Chukchi Borderland pack ice, about 75.4-77.5 N, 160-170 W.
The two NISAR passes are 1 day apart and overlap 93 %.
Sentinel-1: no pass covers the area on 28 Oct, so we use 26, 27 and 29 Oct (two tracking pairs
that bracket the NISAR pair). Coverage of the NISAR overlap: 80 %, 89 % and 75 %.
"""
from __future__ import annotations

import netrc
import sys
from pathlib import Path

import asf_search as asf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAX_TOTAL_GB = 20  # safety cap (user approved about 17 GB on 14 Nov 2026)
EARTHDATA_HOST = "urs.earthdata.nasa.gov"

SCENES = [
    # NISAR L2 GCOV, provisional, L-band, VV (frequency A) + HH (frequency B), 5 MHz
    "NISAR_L2_PR_GCOV_034_047_A_045_0505_SVSH_A_20261027T110140_20261027T110225_P05036_N_F_J_001",
    "NISAR_L2_PR_GCOV_034_061_D_046_0505_SVSH_A_20261028T102047_20261028T102122_P05036_N_F_J_001",
    # Sentinel-1 EW GRD medium resolution, C-band, HH+HV
    "S1C_EW_GRDM_1SDH_20261026T174629_20261026T174729_010059_01413E_F4F5",
    "S1C_EW_GRDM_1SDH_20261027T182718_20261027T182818_010074_0141B9_7AA8",
    "S1C_EW_GRDM_1SDH_20261029T181119_20261029T181148_010103_0142B0_965D",
]
# NISAR pair across the ice edge (about half open water, half ice, no land), 72 N 172 W.
# Used to test ice-vs-water classification against the reference maps.
EDGE_SCENES = [
    "NISAR_L2_PR_GCOV_034_049_A_039_0505_SVSH_A_20261027T141811_20261027T141848_P05036_N_F_J_001",
    "NISAR_L2_PR_GCOV_034_060_D_051_0505_SVSH_A_20261028T084333_20261028T084410_P05036_N_F_J_001",
]
WANTED_LEVELS = ("GCOV", "GRD_MD")


def size_bytes(product) -> int:
    """Size of the main data file. NISAR lists one size per file; Sentinel-1 gives one number."""
    b = product.properties["bytes"]
    if isinstance(b, dict):
        return b[product.properties["fileName"]]["bytes"]
    return int(b)


def find(scenes=SCENES):
    found = [r for r in asf.granule_search(scenes) if r.properties["processingLevel"] in WANTED_LEVELS]
    names = {r.properties["sceneName"] for r in found}
    missing = set(scenes) - names
    if missing:
        raise RuntimeError(f"Not found on ASF: {sorted(missing)}")
    if len(found) != len(scenes):
        raise RuntimeError(f"Expected {len(scenes)} products, ASF returned {len(found)}")
    return found


def dataset_scenes() -> list[str]:
    import json
    d = json.loads((ROOT / "config" / "dataset.json").read_text())
    return [c["name"] for split in ("train", "test") for c in d[split]]


def main(dry_run: bool, with_dataset: bool = False) -> None:
    wanted = SCENES + EDGE_SCENES + (dataset_scenes() if with_dataset else [])
    products = find(list(dict.fromkeys(wanted)))
    total = sum(size_bytes(p) for p in products)
    for p in products:
        print(f"{size_bytes(p) / 1e6:7.0f} MB  {p.properties['fileName']}")
    print(f"{total / 1e9:7.2f} GB  total")
    if total > MAX_TOTAL_GB * 1e9:
        raise RuntimeError(f"Total is above the {MAX_TOTAL_GB} GB cap")
    if dry_run:
        return
    if netrc.netrc().authenticators(EARTHDATA_HOST) is None:  # raises if ~/.netrc is missing
        raise RuntimeError(f"No {EARTHDATA_HOST} entry in ~/.netrc")
    DATA.mkdir(exist_ok=True)
    session = asf.ASFSession()  # reads ~/.netrc
    for p in products:
        if (DATA / p.properties["fileName"]).exists():
            continue
        p.download(path=str(DATA), session=session, fileType=asf.FileDownloadType.DEFAULT_FILE)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv, with_dataset="--dataset" in sys.argv)
