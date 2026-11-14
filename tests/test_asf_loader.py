"""Tests for scripts/asf_loader.py that need no download: a fake extracted .SAFE folder."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.asf_loader import ENL_EW_GRDM, load_asf_item

SCENE = "S1C_EW_GRDM_1SDH_20261027T182718_20261027T182818_010074_0141B9_7AA8"


def make_safe(data_dir, pols=("hh", "hv")):
    safe = data_dir / f"{SCENE}.SAFE"
    for pol in pols:
        stem = f"s1c-ew-grd-{pol}-20261027t182718-001"
        for rel in (f"measurement/{stem}.tiff", f"annotation/{stem}.xml",
                    f"annotation/calibration/calibration-{stem}.xml",
                    f"annotation/calibration/noise-{stem}.xml"):
            f = safe / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
    (data_dir / f"{SCENE}.zip").write_text("already downloaded")  # skips the download step
    return safe


def test_item_has_the_assets_the_notebook_uses(tmp_path):
    make_safe(tmp_path)
    item = load_asf_item(SCENE, tmp_path)
    assert item.id == SCENE
    assert item.properties["sar:looks_equivalent_number"] == ENL_EW_GRDM
    for pol in ("hh", "hv"):
        for key in (pol, f"schema-calibration-{pol}", f"schema-noise-{pol}", f"annotation-{pol}"):
            href = item.assets[key].href
            assert href.startswith("file://") and f"-{pol}-" in href
    assert item.assets["annotation-hh"].href.endswith("001.xml")
    assert "calibration" not in item.assets["annotation-hh"].href


def test_missing_polarisation_fails_fast(tmp_path):
    make_safe(tmp_path, pols=("hh",))
    with pytest.raises(FileNotFoundError):
        load_asf_item(SCENE, tmp_path)
