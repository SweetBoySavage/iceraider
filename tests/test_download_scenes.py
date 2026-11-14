import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_scenes import size_bytes


def test_size_for_sentinel1_is_a_plain_number():
    assert size_bytes(SimpleNamespace(properties={"bytes": 240_000_000, "fileName": "a.zip"})) == 240_000_000


def test_size_for_nisar_uses_the_h5_file_only():
    props = {"fileName": "a.h5", "bytes": {"a.h5": {"bytes": 478, "format": "HDF5"}, "a_QA_STATS.h5": {"bytes": 5}}}
    assert size_bytes(SimpleNamespace(properties=props)) == 478
