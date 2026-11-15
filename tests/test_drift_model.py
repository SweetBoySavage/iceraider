import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.drift_model import (DriftModel, east_north_to_map, map_to_east_north, vector_errors,
                                 wind_from_speed_direction)


def test_north_points_at_the_pole_on_the_map():
    # at 165 W the pole (map origin) lies towards +x, -y: a point there sits at (sin(-120), -cos(-120))
    north = east_north_to_map(0.0, 1.0, -165.0)
    assert north.real == pytest.approx(np.sqrt(3) / 2) and north.imag == pytest.approx(-0.5)
    # on the -45 meridian north is straight up (+y), east is +x
    assert east_north_to_map(0.0, 1.0, -45.0) == pytest.approx(1j)
    assert east_north_to_map(1.0, 0.0, -45.0) == pytest.approx(1.0)
    e, n = map_to_east_north(east_north_to_map(3.0, -2.0, 170.0), 170.0)
    assert (e, n) == (pytest.approx(3.0), pytest.approx(-2.0))


def test_wind_direction_is_where_it_comes_from():
    e, n = wind_from_speed_direction(10.0, 270.0)      # westerly wind blows towards the east
    assert e == pytest.approx(10.0) and n == pytest.approx(0.0, abs=1e-9)
    e, n = wind_from_speed_direction(5.0, 0.0)          # northerly wind blows towards the south
    assert e == pytest.approx(0.0, abs=1e-9) and n == pytest.approx(-5.0)


def test_fit_recovers_known_coefficients_and_forecasts_step_by_step():
    rng = np.random.default_rng(1)
    n = 2000
    v_prev = rng.normal(size=n) + 1j * rng.normal(size=n)
    w_now = 8 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    w_prev = 8 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    a, b, c, d = 0.3, 1.7 * np.exp(-1j * np.deg2rad(30)), -0.2 + 0.1j, 0.5 - 0.2j
    v_next = a * v_prev + b * w_now + c * w_prev + d + 0.05 * rng.normal(size=n)
    m = DriftModel.fit(v_next, v_prev, w_now, w_prev)
    assert np.allclose(m.coef, [a, b, c, d], atol=0.02)
    info = m.describe()["wind now"]
    assert info["turn_right_deg"] == pytest.approx(30, abs=1) and info["percent_of_wind"] == pytest.approx(1.97, abs=0.05)
    steps = m.forecast(v_prev[:3], [w_prev[:3], w_now[:3], w_now[:3]])
    assert len(steps) == 2 and np.allclose(steps[0], m.predict(v_prev[:3], w_now[:3], w_prev[:3]))
    assert np.allclose(steps[1], m.predict(steps[0], w_now[:3], w_now[:3]))


def test_fit_ignores_missing_values_and_reduced_terms_work():
    rng = np.random.default_rng(2)
    w = 5 * (rng.normal(size=500) + 1j * rng.normal(size=500))
    v = 1.5 * w
    v[::10] = np.nan
    m = DriftModel.fit(v, np.zeros(500), w, np.zeros(500), terms=("wind now",))
    assert m.coef[0] == pytest.approx(1.5)


def test_vector_errors():
    obs = np.array([1 + 0j, 0 + 2j, np.nan])
    s = vector_errors(np.array([1 + 1j, 0 + 2j, 5]), obs)
    assert s["n"] == 2 and s["rmse"] == pytest.approx(np.sqrt(0.5))
    assert s["median_abs_angle_deg"] == pytest.approx(22.5)


def test_window_mean_uses_hours_from_start_up_to_end():
    from scripts.wind import window_mean
    times = np.arange("2026-10-27T00", "2026-10-28T00", dtype="datetime64[h]")
    values = np.arange(24.0)[None, :].repeat(2, axis=0)
    assert window_mean(times, values, "2026-10-27T10", "2026-10-27T13").tolist() == [11.0, 11.0]   # hours 10, 11, 12
    with pytest.raises(ValueError):
        window_mean(times, values, "2026-11-01T00", "2026-11-02T00")


def test_window_mean_complete_rejects_missing_hours():
    from scripts.wind import window_mean
    times = np.arange("2026-10-27T00", "2026-10-27T12", dtype="datetime64[h]")
    values = np.ones((1, 12)); values[0, 5] = np.nan
    assert window_mean(times, values, "2026-10-27T00", "2026-10-27T12")[0] == 1.0      # lenient: skips the gap
    with pytest.raises(ValueError):
        window_mean(times, values, "2026-10-27T00", "2026-10-27T12", complete=True)    # hour 5 missing
    with pytest.raises(ValueError):
        window_mean(times, np.ones((1, 12)), "2026-10-27T06", "2026-10-28T06", complete=True)  # runs past the data
