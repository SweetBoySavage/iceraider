"""Build the training table for the drift model: OSI SAF ice drift + ERA5 wind, Chukchi / western Beaufort Sea.

    python scripts/build_drift_training.py        # writes data/drift_training.npz

Ice drift: OSI SAF low-resolution sea-ice drift (OSI-405, "merged" multi-sensor product), 62.5 km grid,
one 48 h displacement per day (12:00 UTC to 12:00 UTC two days later). Free, from thredds.met.no.
Only cells flagged "nominal quality" (status 30) are used.
Wind: hourly ERA5 10 m wind (Open-Meteo), on every second grid cell (125 km), averaged over the same 48 h.

One sample = one grid cell and two back-to-back 48 h windows (previous, next):
    v_prev, v_next  ice velocity (km/day, complex, map frame)
    w_prev, w_next  mean wind (m/s, complex, map frame)
Seasons: 15 Sep - 30 Nov, 2018-2026 (2026 up to the newest file). Windows start every second day.
Only the Chukchi box is kept from each file (data/osisaf/region/*.npz); full files are deleted.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.drift_model import east_north_to_map            # noqa: E402
from scripts.wind import fetch_hourly, window_mean            # noqa: E402

OSI = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/drift_lr/merged"
HEADERS = {"User-Agent": "sea-ice-drift-research-script"}
REGION_DIR = ROOT / "data" / "osisaf" / "region"
OUT = ROOT / "data" / "drift_training.npz"
BOX = dict(lat=(70.0, 80.0), lon=(-180.0, -150.0))       # Chukchi + western Beaufort Sea
YEARS = range(2018, 2027)
SEASON = ((9, 15), (11, 30))
NOMINAL = 30


def file_name(start: date) -> str:
    end = start + timedelta(days=2)
    return f"ice_drift_nh_polstere-625_multi-oi_{start:%Y%m%d}1200-{end:%Y%m%d}1200.nc"


def region_slices(ds):
    lat, lon = ds.lat.values, ds.lon.values
    box = (lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1]) & (lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1])
    r, c = np.where(box)
    return slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1), box[r.min():r.max() + 1, c.min():c.max() + 1]


def fetch_region(start: date, rows=None, cols=None):
    """Region subset of one OSI SAF file as a dict of arrays, cached; None if the file does not exist."""
    out = REGION_DIR / f"{start:%Y%m%d}.npz"
    if out.exists():
        z = np.load(out)
        return {k: z[k] for k in z.files} if z.files else None
    REGION_DIR.mkdir(parents=True, exist_ok=True)
    end = start + timedelta(days=2)
    url = f"{OSI}/{end:%Y}/{end:%m}/{file_name(start)}"
    tmp = REGION_DIR / "_download.nc"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            np.savez(out)                                  # remember that it is missing
            return None
        raise
    with xr.open_dataset(tmp, decode_timedelta=False) as ds:
        if rows is None:
            rows, cols, _ = region_slices(ds)
        sub = ds.isel(yc=rows, xc=cols)
        data = dict(dX=sub.dX.values[0].astype("float64"), dY=sub.dY.values[0].astype("float64"),
                    flag=sub.status_flag.values[0].astype("int16"),
                    dt0=sub.dt0.values[0].astype("float64"), dt1=sub.dt1.values[0].astype("float64"),
                    lat=sub.lat.values.astype("float64"), lon=sub.lon.values.astype("float64"))
    tmp.unlink()
    np.savez(out, **data)
    return data


def season_dates(year: int, last: date):
    d, end = date(year, *SEASON[0]), min(date(year, *SEASON[1]), last)
    while d <= end:
        yield d
        d += timedelta(days=2)


def main():
    last = date.today() - timedelta(days=3)                # newest complete 48 h window
    rows = cols = None
    fields = {}
    for year in YEARS:
        got = 0
        for d in season_dates(year, last):
            f = fetch_region(d, rows, cols)
            if f is not None:
                fields[d] = f
                got += 1
        print(f"{year}: {got} OSI SAF files", flush=True)
    any_field = next(iter(fields.values()))
    lat, lon = any_field["lat"], any_field["lon"]
    in_box = (lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1]) & (lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1])

    # wind points: every second cell inside the box; each cell uses the nearest wind point
    wr, wc = np.where(in_box & (np.arange(lat.shape[0])[:, None] % 2 == 0) & (np.arange(lat.shape[1])[None, :] % 2 == 0))
    rr, cc = np.indices(lat.shape)
    nearest = np.argmin((rr[..., None] - wr) ** 2 + (cc[..., None] - wc) ** 2, axis=-1)
    print(f"{in_box.sum()} cells in the box, {len(wr)} wind points", flush=True)

    rows_out = {k: [] for k in ("v_prev", "v_next", "w_prev", "w_next", "year", "day", "lat", "lon")}
    for year in YEARS:
        days = sorted(d for d in fields if d.year == year)
        if len(days) < 2:
            continue
        t, east, north = fetch_hourly(lat[wr, wc], lon[wr, wc], f"{days[0]:%Y-%m-%d}", f"{days[-1] + timedelta(days=3):%Y-%m-%d}")
        wind_map = east_north_to_map(east, north, lon[wr, wc][:, None])          # (n_points, n_hours)
        for d0, d1 in zip(days, days[1:]):
            if d1 - d0 != timedelta(days=2):
                continue
            A, B = fields[d0], fields[d1]
            ok = in_box & (A["flag"] == NOMINAL) & (B["flag"] == NOMINAL)
            if not ok.any():
                continue
            hours = lambda F: 48 + np.nan_to_num(F["dt1"] - F["dt0"]) / 3600       # noqa: E731
            v = {k: (F["dX"] + 1j * F["dY"]) / (hours(F) / 24) for k, F in (("prev", A), ("next", B))}
            w_prev = window_mean(t, wind_map, f"{d0}T12", f"{d0 + timedelta(days=2)}T12")[nearest]
            w_next = window_mean(t, wind_map, f"{d1}T12", f"{d1 + timedelta(days=2)}T12")[nearest]
            for k, val in (("v_prev", v["prev"]), ("v_next", v["next"]), ("w_prev", w_prev), ("w_next", w_next),
                           ("lat", lat), ("lon", lon)):
                rows_out[k].append(val[ok])
            rows_out["year"].append(np.full(ok.sum(), year))
            rows_out["day"].append(np.full(ok.sum(), np.datetime64(d1, "D")))
        print(f"{year}: samples so far {sum(map(len, rows_out['year']))}", flush=True)
    out = {k: np.concatenate(v) for k, v in rows_out.items()}
    np.savez(OUT, **out)
    print(f"wrote {OUT} with {len(out['year'])} samples")


if __name__ == "__main__":
    main()
