"""Fit and test the drift model on OSI SAF drift + ERA5 wind (run build_drift_training.py first).

    python scripts/train_drift_model.py      # prints the comparison, writes models/drift_model.json

Fair test: fit on the 2018-2023 autumns, score on 2024-2026, which the fit never saw. Compared with
    zero          no motion
    persistence   tomorrow's drift = today's drift
    wind only     V = b * W[k] + d            (classic free-drift rule, fitted)
    full model    V = a*V[k-1] + b*W[k] + c*W[k-1] + d
The model used for forecasts is then refitted on 2018-2025 (2026 holds the NISAR test days, so it is
never used for fitting).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.drift_model import TERMS, DriftModel, vector_errors      # noqa: E402

DATA = ROOT / "data" / "drift_training.npz"
OUT = ROOT / "models" / "drift_model.json"
TRAIN, TEST, FINAL = range(2018, 2024), range(2024, 2027), range(2018, 2026)
WIND_ONLY = ("wind now", "constant")


def load():
    z = np.load(DATA)
    return {k: z[k] for k in z.files}


def subset(d, years):
    sel = np.isin(d["year"], list(years))
    return {k: v[sel] for k, v in d.items()}


def scores(d, models):
    out = {"zero": vector_errors(np.zeros_like(d["v_next"]), d["v_next"]),
           "persistence": vector_errors(d["v_prev"], d["v_next"])}
    for name, m in models.items():
        out[name] = vector_errors(m.predict(d["v_prev"], d["w_next"], d["w_prev"]), d["v_next"])
    return out


def fit_both(d):
    return {"wind only": DriftModel.fit(d["v_next"], d["v_prev"], d["w_next"], d["w_prev"], WIND_ONLY),
            "full model": DriftModel.fit(d["v_next"], d["v_prev"], d["w_next"], d["w_prev"], TERMS)}


def main():
    d = load()
    train, test = subset(d, TRAIN), subset(d, TEST)
    print(f"samples: train {len(train['year'])} ({TRAIN.start}-{TRAIN.stop - 1}), test {len(test['year'])} "
          f"({TEST.start}-{TEST.stop - 1}); mean observed speed in test {np.abs(test['v_next']).mean():.2f} km/day")
    models = fit_both(train)
    res = scores(test, models)
    print(f"\n{'method':12s} {'RMSE km/day':>11s} {'vs persistence':>15s} {'speed bias':>11s} {'median angle error':>19s}")
    for name, s in res.items():
        print(f"{name:12s} {s['rmse']:11.2f} {100 * (1 - s['rmse'] / res['persistence']['rmse']):14.0f}% "
              f"{s['speed_bias']:11.2f} {s['median_abs_angle_deg']:18.0f}°")
    for year in TEST:
        t = subset(test, [year])
        if len(t["year"]):
            r = scores(t, models)
            print(f"  {year}: n={r['zero']['n']:5d}  persistence {r['persistence']['rmse']:.2f}  "
                  f"wind only {r['wind only']['rmse']:.2f}  full {r['full model']['rmse']:.2f} km/day")

    final = fit_both(subset(d, FINAL))
    for name, m in final.items():
        print(f"\n{name} (fitted on {FINAL.start}-{FINAL.stop - 1}):")
        for term, info in m.describe().items():
            print(f"  {term:15s} {info}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "terms": list(TERMS), "coef_real": final["full model"].coef.real.tolist(), "coef_imag": final["full model"].coef.imag.tolist(),
        "wind_only_terms": list(WIND_ONLY), "wind_only_real": final["wind only"].coef.real.tolist(),
        "wind_only_imag": final["wind only"].coef.imag.tolist(),
        "fitted_years": [FINAL.start, FINAL.stop - 1], "units": "velocity km/day, wind m/s, map frame EPSG:3413",
        "test": {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in s.items()} for k, s in res.items()},
        "describe": final["full model"].describe()}, indent=2))
    print(f"\nwrote {OUT}")


def load_models(path=OUT):
    j = json.loads(Path(path).read_text())
    full = DriftModel(tuple(j["terms"]), np.array(j["coef_real"]) + 1j * np.array(j["coef_imag"]))
    wind = DriftModel(tuple(j["wind_only_terms"]), np.array(j["wind_only_real"]) + 1j * np.array(j["wind_only_imag"]))
    return full, wind


if __name__ == "__main__":
    main()
