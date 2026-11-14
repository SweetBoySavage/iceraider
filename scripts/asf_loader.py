"""Load a Sentinel-1 GRD scene from NASA ASF DAAC instead of Microsoft Planetary Computer.

Returns an `item` object that looks like the Planetary Computer STAC item the notebook uses,
so most notebook cells run unchanged.

Setup (once):
    pip install asf_search
    Create a free NASA Earthdata Login: https://urs.earthdata.nasa.gov
    Put it in ~/.netrc:  machine urs.earthdata.nasa.gov login <user> password <pass>

SCENE_ID must be the full ASF name, including the 4-character ending (e.g. ..._0685B0_1A2B).
The Planetary Computer id leaves that ending out, and ASF will not find the scene without it.

Notebook changes:
    1. Replace the pystac_client / planetary_computer cell with:
           from scripts.asf_loader import load_asf_item
           item = load_asf_item(SCENE_ID, DATA)
    2. In the incidence-angle cell, replace the `url = ...replace(...)` line with:
           url = item.assets["annotation-hh"].href
"""
from __future__ import annotations

import netrc
import zipfile
from pathlib import Path
from types import SimpleNamespace

# Equivalent Number of Looks for EW GRD medium resolution (ESA product spec value).
# The Planetary Computer item provided this as a property; ASF does not, so it is set here.
ENL_EW_GRDM = 10.7
EARTHDATA_HOST = "urs.earthdata.nasa.gov"


def _download(scene_id: str, data_dir: Path) -> Path:
    import asf_search as asf

    zip_path = data_dir / f"{scene_id}.zip"
    if zip_path.exists():
        return zip_path
    results = [r for r in asf.granule_search([scene_id])
               if r.properties.get("processingLevel", "").startswith("GRD")]
    if not results:
        raise RuntimeError(f"No GRD product found on ASF for {scene_id}. Use the full ASF name (with the 4-character ending).")
    if netrc.netrc().authenticators(EARTHDATA_HOST) is None:  # raises if ~/.netrc is missing
        raise RuntimeError(f"No {EARTHDATA_HOST} entry in ~/.netrc")
    results[0].download(path=str(data_dir), session=asf.ASFSession())  # session reads ~/.netrc; ~250 MB for EW GRDM
    return zip_path


def _extract(zip_path: Path, data_dir: Path) -> Path:
    safe_dir = data_dir / f"{zip_path.stem}.SAFE"
    if not safe_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
    return safe_dir


def _one(safe: Path, pattern: str) -> Path:
    hits = sorted(safe.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 file for {pattern}, found {len(hits)}")
    return hits[0]


def load_asf_item(scene_id: str, data_dir: Path, pols=("hh", "hv")):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    safe = _extract(_download(scene_id, data_dir), data_dir)

    assets = {}
    for pol in pols:
        files = {
            pol: _one(safe, f"measurement/*-{pol}-*.tiff"),
            f"schema-calibration-{pol}": _one(safe, f"annotation/calibration/calibration-*-{pol}-*.xml"),
            f"schema-noise-{pol}": _one(safe, f"annotation/calibration/noise-*-{pol}-*.xml"),
            f"annotation-{pol}": _one(safe, f"annotation/*-{pol}-*.xml"),
        }
        # file:// URIs work with both rasterio.open and urllib.request.urlretrieve used in the notebook
        assets.update({k: SimpleNamespace(href=p.resolve().as_uri()) for k, p in files.items()})

    return SimpleNamespace(
        id=scene_id,
        assets=assets,
        properties={"sar:looks_equivalent_number": ENL_EW_GRDM},
    )
