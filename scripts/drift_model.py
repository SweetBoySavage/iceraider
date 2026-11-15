"""Predict sea-ice drift from the previous drift and the wind: a small linear model with complex coefficients.

Vectors are complex numbers on the polar-stereographic map grid (x + i*y, EPSG:3413 orientation): velocity
in km/day, wind in m/s. For one ice parcel and one time step k:

    V[k] = a * V[k-1]  +  b * W[k]  +  c * W[k-1]  +  d

    V[k-1]  drift during the previous step (km/day)
    W[k]    mean 10 m wind during the step being predicted (m/s; a forecast when used for real)
    W[k-1]  mean 10 m wind during the previous step
    a,b,c,d complex coefficients, fitted by least squares

A complex coefficient both scales and turns a vector: |b| is the ice speed per m/s of wind, and -angle(b)
is how far the ice moves to the RIGHT of the wind (free drift: about 2 % of the wind speed, 20-40 degrees
to the right in the Northern Hemisphere). `a` is the part of the previous drift that persists (ocean
currents, slow changes), `c` corrects for the wind that drove the previous drift, `d` is a mean drift.

Longer forecasts are made step by step: the predicted V[k] becomes V[k-1] for the next step.
Full description: docs/drift_forecast.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KM_PER_DAY_PER_M_S = 86.4  # 1 m/s = 86.4 km/day
TERMS = ("previous drift", "wind now", "wind before", "constant")


def east_north_to_map(east, north, lon_deg):
    """Rotate east/north components into the polar-stereographic map frame (central meridian -45 deg).

    In that frame the local east direction points at angle (lon + 45 deg) from the map x axis, and north
    is 90 deg counter-clockwise from east. Returns a complex number x + i*y.
    """
    return (np.asarray(east) + 1j * np.asarray(north)) * np.exp(1j * np.deg2rad(np.asarray(lon_deg) + 45.0))


def map_to_east_north(z, lon_deg):
    """Inverse of east_north_to_map: (east, north) components of a complex map vector."""
    en = np.asarray(z) * np.exp(-1j * np.deg2rad(np.asarray(lon_deg) + 45.0))
    return en.real, en.imag


def wind_from_speed_direction(speed, direction_deg):
    """(east, north) wind components from speed and meteorological direction (where the wind comes FROM)."""
    rad = np.deg2rad(np.asarray(direction_deg, dtype="float64"))
    speed = np.asarray(speed, dtype="float64")
    return -speed * np.sin(rad), -speed * np.cos(rad)


def design(v_prev, w_now, w_prev, terms=TERMS):
    """Complex design matrix with one column per term in `terms`."""
    cols = {"previous drift": v_prev, "wind now": w_now, "wind before": w_prev,
            "constant": np.ones_like(np.asarray(w_now, dtype="complex128"))}
    return np.column_stack([np.asarray(cols[t], dtype="complex128").ravel() for t in terms])


@dataclass
class DriftModel:
    terms: tuple
    coef: np.ndarray            # complex, one per term

    @classmethod
    def fit(cls, v_next, v_prev, w_now, w_prev, terms=TERMS) -> "DriftModel":
        X = design(v_prev, w_now, w_prev, terms)
        y = np.asarray(v_next, dtype="complex128").ravel()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        return cls(tuple(terms), coef)

    def predict(self, v_prev, w_now, w_prev):
        shape = np.shape(w_now)
        return (design(v_prev, w_now, w_prev, self.terms) @ self.coef).reshape(shape)

    def forecast(self, v0, winds):
        """Step-by-step forecast. v0: drift of the last observed step; winds: [W0, W1, ..., Wn] where W0 is
        the wind during the observed step. Returns [V1, ..., Vn] (each step uses the previous prediction)."""
        out, v_prev = [], v0
        for k in range(1, len(winds)):
            v_prev = self.predict(v_prev, winds[k], winds[k - 1])
            out.append(v_prev)
        return out

    def describe(self) -> dict:
        """Coefficients in readable units: scale and turning angle (positive = to the right)."""
        out = {}
        for t, c in zip(self.terms, self.coef):
            turn = float(-np.rad2deg(np.angle(c)))
            if t.startswith("wind"):
                out[t] = dict(percent_of_wind=round(float(100 * abs(c) / KM_PER_DAY_PER_M_S), 2), turn_right_deg=round(turn, 1))
            elif t == "constant":
                out[t] = dict(km_per_day=round(float(abs(c)), 2), map_angle_deg=round(float(np.rad2deg(np.angle(c))), 1))
            else:
                out[t] = dict(factor=round(float(abs(c)), 3), turn_right_deg=round(turn, 1))
        return out


def vector_errors(pred, obs) -> dict:
    """Vector error statistics (same units as the inputs)."""
    pred, obs = np.asarray(pred).ravel(), np.asarray(obs).ravel()
    ok = np.isfinite(pred) & np.isfinite(obs)
    err = pred[ok] - obs[ok]
    speed_obs = np.abs(obs[ok])
    angle = np.rad2deg(np.angle(pred[ok] / obs[ok]))
    return dict(n=int(ok.sum()), rmse=float(np.sqrt(np.mean(np.abs(err) ** 2))),
                mean_obs_speed=float(speed_obs.mean()) if ok.any() else float("nan"),
                speed_bias=float(np.mean(np.abs(pred[ok]) - speed_obs)) if ok.any() else float("nan"),
                median_abs_angle_deg=float(np.median(np.abs(angle))) if ok.any() else float("nan"))
