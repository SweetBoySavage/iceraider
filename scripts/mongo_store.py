"""Store scene info, classification results and ice-motion vectors in MongoDB (database NasaData).

Images are NOT stored (16 MB document limit): only file paths.
The connection string comes from the environment variable MONGODB_URI, or from a .env file
in the repo root (git-ignored). It is never printed.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Geod, Transformer

DB_NAME = "NasaData"
GEO_INDEXES = {"scenes": "footprint", "ice_motion": "location"}


def mongo_uri(env_file: Path = Path(__file__).resolve().parents[1] / ".env") -> str:
    uri = os.environ.get("MONGODB_URI")
    if not uri and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("MONGODB_URI="):
                uri = line.split("=", 1)[1].strip().strip("'\"")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set (environment variable or .env file)")
    return uri


def get_db():
    from pymongo import GEOSPHERE, MongoClient
    db = MongoClient(mongo_uri(), serverSelectionTimeoutMS=8000)[DB_NAME]
    for collection, field in GEO_INDEXES.items():
        db[collection].create_index([(field, GEOSPHERE)])
    return db


def scene_doc(name, sensor, product, start: datetime, file_path, footprint, polarisations, band, projection):
    if sensor not in ("NISAR", "Sentinel-1"):
        raise ValueError(f"unknown sensor {sensor}")
    return {"name": name, "sensor": sensor, "product": product, "datetime": start, "file_path": str(file_path),
            "footprint": footprint, "polarisations": list(polarisations), "frequency_band": band,
            "projection": projection}


def result_doc(scene_name, method, channels, class_shares, threshold_db, dark_pct, window, note):
    return {"scene": scene_name, "method": method, "channels": list(channels),
            "class_shares_pct": [round(100 * s, 2) for s in class_shares],
            "threshold_db": round(threshold_db, 2), "dark_pct": round(100 * dark_pct, 2),
            "bright_ice_pct": round(100 * (1 - dark_pct), 2), "window_row_col_size": list(window), "note": note}


def motion_docs(track, x, y, epsg, spacing_m, start: datetime, hours, source, scene_from, scene_to):
    """One GeoJSON point per tracked patch: where the ice was at `start`, how fast and where to it moved."""
    rows, cols = track["row"].astype(int), track["col"].astype(int)
    x0, y0 = x[cols], y[rows]
    x1, y1 = x0 + track["d_col"] * spacing_m, y0 - track["d_row"] * spacing_m      # y decreases with row
    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon0, lat0 = to_lonlat.transform(x0, y0)
    lon1, lat1 = to_lonlat.transform(x1, y1)
    bearing, _, dist_m = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)
    end = start + timedelta(hours=hours)
    return [{"location": {"type": "Point", "coordinates": [float(lon0[i]), float(lat0[i])]},
             "date": start, "date_end": end, "hours": hours,
             "displacement_km": round(float(dist_m[i]) / 1000, 3),
             "speed_km_day": round(float(dist_m[i]) / 1000 / hours * 24, 3),
             "direction_deg": round(float(bearing[i]) % 360, 1),      # compass direction the ice moves TO
             "similarity": round(float(track["similarity"][i]), 3),
             "source": source, "scene_from": scene_from, "scene_to": scene_to}
            for i in range(len(rows))]


def replace_many(collection, key: dict, docs: list[dict]) -> int:
    """Replace all documents matching `key` by `docs`, so re-running never makes duplicates."""
    collection.delete_many(key)
    if docs:
        collection.insert_many(docs)
    return len(docs)
