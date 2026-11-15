"""Start the route interface. One command, then the browser opens:

    python scripts/run_app.py            # options: --port 8000, --no-browser

Needs only the files in app/ (in git) and numpy + pyproj. The page asks this server whether a route is safe,
so the rules live in one place: scripts/route_safety.py.
"""
from __future__ import annotations

import argparse
import json
import sys
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
MAX_ROUTE_POINTS = 200


def load_bundle(path: Path | None = None) -> dict:
    """The map the safety check reads (app/data/bundle.npz, made by scripts/build_app_data.py)."""
    path = Path(path) if path else APP_DATA / "bundle.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/build_mosaic.py, then python scripts/build_app_data.py")
    with np.load(path) as f:
        out = {k: f[k] for k in ARRAYS}
        out |= {k: float(f[k]) for k in ("x0", "y0", "cell_m", "margin_m")}
        out |= {"epsg": int(f["epsg"]), "scenes": [str(s) for s in f["scenes"]], "skipped": [str(s) for s in f["skipped"]]}
    return out


def parse_query(query: str) -> dict:
    """Numbers and route from '?ship_m=200&speed_kn=10&ice_buffer_m=1000&coast_margin_m=1000&route=lat,lon;lat,lon'."""
    raw = {k: v[0] for k, v in parse_qs(query).items()}
    try:
        values = {k: float(raw[k]) for k in ("ship_m", "speed_kn", "ice_buffer_m", "coast_margin_m")}
        points = [tuple(float(n) for n in pair.split(",")) for pair in raw["route"].split(";")]
    except KeyError as missing:
        raise ValueError(f"missing value: {missing.args[0]}") from None
    if any(len(p) != 2 for p in points) or not 2 <= len(points) <= MAX_ROUTE_POINTS:
        raise ValueError(f"route must be 2 to {MAX_ROUTE_POINTS} 'lat,lon' pairs separated by ';'")
    return values | {"route": points}


def answer(bundle: dict, query: str) -> dict:
    """The reply to /api/check: the verdict plus every sampled point (map cell position, status, image number)."""
    q = parse_query(query)
    out = route_safety.check_route(bundle, route_safety.to_xy(q["route"], bundle["epsg"]),
                                   q["ship_m"], q["ice_buffer_m"], q["coast_margin_m"])
    verdict = route_safety.verdict(out["dist_m"], out["status"], q["speed_kn"])
    return {"verdict": verdict,
            "km": np.round(out["dist_m"] / 1e3, 3).tolist(),
            "col": np.round((out["xy"][:, 0] - bundle["x0"]) / bundle["cell_m"], 2).tolist(),
            "row": np.round((bundle["y0"] - out["xy"][:, 1]) / bundle["cell_m"], 2).tolist(),
            "status": out["status"].tolist(), "scene": out["scene"].tolist()}


class Handler(SimpleHTTPRequestHandler):
    bundle: dict = {}

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/api/check":
            return super().do_GET()
        try:
            code, body = 200, answer(self.bundle, url.query)
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


def make_server(port: int, bundle: dict) -> ThreadingHTTPServer:
    for name in ("cells.png", "distance.png", "meta.json"):
        if not (APP_DATA / name).exists():
            raise FileNotFoundError(f"{APP_DATA / name} is missing. Run: python scripts/build_app_data.py")
    handler = type("AppHandler", (Handler,), {"bundle": bundle})
    return ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(APP)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.port, load_bundle())
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
