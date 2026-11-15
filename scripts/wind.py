"""Hourly 10 m wind from the ERA5 reanalysis, via the Open-Meteo archive API (free, no account).

    fetch_hourly(lats, lons, "2026-10-26", "2026-10-31") -> (times, east, north)
        times: datetime64[h] (UTC), east/north: m/s, shape (n_points, n_hours)

Responses are cached in data/wind/, so a rerun does not download again.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from scripts.drift_model import wind_from_speed_direction

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "wind"
URL = "https://archive-api.open-meteo.com/v1/archive"
HEADERS = {"User-Agent": "sea-ice-drift-research-script"}
CHUNK = 20          # locations per request


def _get(params: dict) -> list:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    key = hashlib.sha1(query.encode()).hexdigest()[:16]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{URL}?{query}", headers=HEADERS), timeout=120) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(65)                       # rate limit: wait for the next minute
    data = data if isinstance(data, list) else [data]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    time.sleep(2)
    return data


def fetch_hourly(lats, lons, start_date: str, end_date: str):
    lats, lons = np.round(np.asarray(lats, float), 3), np.round(np.asarray(lons, float), 3)
    east, north, times = [], [], None
    for i in range(0, len(lats), CHUNK):
        params = dict(latitude=",".join(map(str, lats[i:i + CHUNK])), longitude=",".join(map(str, lons[i:i + CHUNK])),
                      start_date=start_date, end_date=end_date, hourly="wind_speed_10m,wind_direction_10m",
                      wind_speed_unit="ms", timezone="GMT", models="era5")
        for loc in _get(params):
            h = loc["hourly"]
            t = np.array(h["time"], dtype="datetime64[h]")
            times = t if times is None else times
            if len(t) != len(times) or (t != times).any():
                raise ValueError("locations returned different time axes")
            spd = np.array([np.nan if v is None else v for v in h["wind_speed_10m"]], float)
            dirn = np.array([np.nan if v is None else v for v in h["wind_direction_10m"]], float)
            e, n = wind_from_speed_direction(spd, dirn)
            east.append(e)
            north.append(n)
    return times, np.array(east), np.array(north)


def window_mean(times, values, start, end, complete: bool = False):
    """Mean over hours with start <= t < end (values: (..., n_hours)). start/end: datetime64.

    complete=True: raise if any hour of the window is missing (ERA5 lags real time by about 5 days), instead
    of averaging the hours that exist."""
    sel = (times >= np.datetime64(start, "h")) & (times < np.datetime64(end, "h"))
    expected = int((np.datetime64(end, "h") - np.datetime64(start, "h")) / np.timedelta64(1, "h"))
    if not sel.any():
        raise ValueError(f"no wind data between {start} and {end}")
    if complete and (sel.sum() < expected or not np.isfinite(values[..., sel]).all()):
        raise ValueError(f"wind data incomplete between {start} and {end}")
    return np.nanmean(values[..., sel], axis=-1)
