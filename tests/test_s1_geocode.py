import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.control import GroundControlPoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.s1_geocode import calibration_table, href_to_path, interpolate_table, warp_to_grid

XML = """<calibration><calibrationVectorList count="2">
<calibrationVector><line>0</line><pixel count="3">0 2 4</pixel><sigmaNought count="3">10 20 30</sigmaNought></calibrationVector>
<calibrationVector><line>4</line><pixel count="3">0 2 4</pixel><sigmaNought count="3">20 30 40</sigmaNought></calibrationVector>
</calibrationVectorList></calibration>"""


def test_calibration_table_is_read_and_interpolated(tmp_path):
    f = tmp_path / "cal.xml"
    f.write_text(XML)
    lines, pixels, values = calibration_table(str(f))
    lut = interpolate_table(lines, pixels, values, n_rows=6, n_cols=5)
    assert lut.shape == (6, 5)
    assert lut[0, 0] == 10 and lut[0, 1] == 15 and lut[2, 2] == 25 and lut[4, 4] == 40
    assert lut[5, 4] == 40        # rows past the last vector keep its value


def test_file_uri_becomes_a_path():
    assert href_to_path("file:///a/b%20c.tiff") == "/a/b c.tiff"
    assert href_to_path("/a/b.tiff") == "/a/b.tiff"


def test_warp_puts_pixels_at_the_right_map_position():
    # 40 m radar-geometry image whose rows run north->south and columns west->east, in EPSG:3413 metres
    img = np.zeros((100, 100), "float32")
    img[20:40, 60:80] = 1.0
    gcps = [GroundControlPoint(row=r, col=c, x=1000 + 40 * c, y=9000 - 40 * r)
            for r in (0, 50, 100) for c in (0, 50, 100)]
    x = np.arange(1040, 5000, 80.0)
    y = np.arange(8960, 5000, -80.0)
    out = warp_to_grid(img, gcps, "EPSG:3413", x, y, 3413)
    rows, cols = np.where(out > 0.5)
    assert x[cols].min() == pytest.approx(1000 + 40 * 60, abs=80) and x[cols].max() == pytest.approx(1000 + 40 * 80, abs=80)
    assert y[rows].max() == pytest.approx(9000 - 40 * 20, abs=80) and y[rows].min() == pytest.approx(9000 - 40 * 40, abs=80)
