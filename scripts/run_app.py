"""Start the route interface. One command, then the browser opens:

    python scripts/run_app.py            # options: --port 8000, --no-browser, --host 0.0.0.0 (show it to other
                                         # computers on your network; default is this computer only)

Needs only the files in app/ (in git) and numpy + pyproj. The page asks this server whether a route is safe,
so the rules live in one place: scripts/route_safety.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import route_safety                                   # noqa: E402

APP = ROOT / "app"
APP_DATA = APP / "data"
ARRAYS = ("classes", "scene_index", "ice_distance_m", "land_distance_m")
DRIFT_ARRAYS = ("u", "v", "observed", "edges_s")
MAX_ROUTE_POINTS = 200


def utc_seconds(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def load_drift(path: Path) -> dict | None:
    """The ice drift forecast of a scenario (app/data/<id>/drift.npz, made by scripts/build_app_drift.py), or None."""
    if not Path(path).exists():
        return None
    with np.load(path) as f:
        out = {k: f[k] for k in DRIFT_ARRAYS}
        out |= {k: float(f[k]) for k in ("x0", "y0", "node_m", "horizon_s")}
    return out


def load_bundle(path: Path) -> dict:
    """The map the safety check reads for one scenario (app/data/<id>/bundle.npz, made by scripts/build_app_data.py)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/build_mosaic.py, then python scripts/build_app_data.py")
    with np.load(path) as f:
        out = {k: f[k] for k in ARRAYS}
        out |= {k: float(f[k]) for k in ("x0", "y0", "cell_m", "margin_m")}
        out |= {"epsg": int(f["epsg"]), "scenes": [str(s) for s in f["scenes"]], "skipped": [str(s) for s in f["skipped"]]}
    return out


def load_bundles(app_data: Path | None = None) -> dict[str, dict]:
    """Every scenario listed in app/data/scenarios.json, by id. Refuses to start when a file of the page is missing."""
    app_data = Path(app_data) if app_data else APP_DATA
    index = app_data / "scenarios.json"
    if not index.exists():
        raise FileNotFoundError(f"{index} is missing. Run: python scripts/build_app_data.py")
    bundles = {}
    for scenario in json.loads(index.read_text()):
        folder = app_data / scenario["id"]
        for name in ("cells.png", "distance.png", "meta.json"):
            if not (folder / name).exists():
                raise FileNotFoundError(f"{folder / name} is missing. Run: python scripts/build_app_data.py")
        bundle = load_bundle(folder / "bundle.npz")
        meta = json.loads((folder / "meta.json").read_text())
        bundle["departure_s"] = utc_seconds(meta["route"]["departure_utc"])
        bundle["scene_time_s"] = np.array([utc_seconds(sc["time_utc"]) for sc in meta["scenes"]])
        bundle["drift"] = load_drift(folder / "drift.npz")
        bundles[scenario["id"]] = bundle
    return bundles


def parse_query(query: str) -> dict:
    """Scenario, numbers and route from
    '?scenario=oct&ship_m=200&speed_kn=10&ice_buffer_m=1000&coast_margin_m=1000&route=lat,lon;lat,lon'
    plus, optional, '&delay_h=24' (departure later than the scenario's, hours) and '&moving=0' (static ice)."""
    raw = {k: v[0] for k, v in parse_qs(query).items()}
    try:
        values = {k: float(raw[k]) for k in ("ship_m", "speed_kn", "ice_buffer_m", "coast_margin_m")}
        values["delay_h"] = float(raw.get("delay_h", 0))
        values["moving"] = raw.get("moving", "1") not in ("0", "false")
        points = [tuple(float(n) for n in pair.split(",")) for pair in raw["route"].split(";")]
        scenario = raw["scenario"]
    except KeyError as missing:
        raise ValueError(f"missing value: {missing.args[0]}") from None
    if any(len(p) != 2 for p in points) or not 2 <= len(points) <= MAX_ROUTE_POINTS:
        raise ValueError(f"route must be 2 to {MAX_ROUTE_POINTS} 'lat,lon' pairs separated by ';'")
    return values | {"route": points, "scenario": scenario}


def answer(bundles: dict[str, dict], query: str) -> dict:
    """The reply to /api/check: the verdict plus every sampled point (map cell position, status, image number)."""
    q = parse_query(query)
    if q["scenario"] not in bundles:
        raise ValueError(f"unknown scenario {q['scenario']!r}; there are: {sorted(bundles)}")
    bundle = bundles[q["scenario"]]
    route_safety.check_inputs(delay_h=q["delay_h"])
    depart_s = bundle.get("departure_s", 0.0) + q["delay_h"] * 3600
    moving = q["moving"] and bundle.get("drift") is not None
    out = route_safety.check_route(bundle, route_safety.to_xy(q["route"], bundle["epsg"]),
                                   q["ship_m"], q["ice_buffer_m"], q["coast_margin_m"],
                                   speed_kn=q["speed_kn"], depart_s=depart_s, moving=moving)
    verdict = route_safety.verdict(out["dist_m"], out["status"], q["speed_kn"])
    drift = bundle.get("drift")
    return {"verdict": verdict, "moving": moving, "delay_h": q["delay_h"],
            "forecast_end_h": None if drift is None else (drift["horizon_s"] - bundle["departure_s"]) / 3600,
            "km": np.round(out["dist_m"] / 1e3, 3).tolist(),
            "col": np.round((out["xy"][:, 0] - bundle["x0"]) / bundle["cell_m"], 2).tolist(),
            "row": np.round((bundle["y0"] - out["xy"][:, 1]) / bundle["cell_m"], 2).tolist(),
            "status": out["status"].tolist(), "scene": out["scene"].tolist()}


class Handler(SimpleHTTPRequestHandler):
    bundles: dict = {}

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/api/check":
            return super().do_GET()
        try:
            code, body = 200, answer(self.bundles, url.query)
        except ValueError as error:
            code, body = 400, {"error": str(error)}
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")          # a presentation must never show an old map
        super().end_headers()

    def log_message(self, *args):
        pass


def make_server(port: int, bundles: dict[str, dict], host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("AppHandler", (Handler,), {"bundles": bundles})
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(APP)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets other computers on the network open the page")
    args = parser.parse_args()
    server = make_server(args.port, load_bundles(), args.host)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Route interface: {url}   (stop with Ctrl+C)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
