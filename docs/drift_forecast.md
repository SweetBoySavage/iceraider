# Forecasting the ice drift arrows

Given the ice drift measured by NISAR on one day and the wind, where will the same ice go on the next days?
This adds a small, physically interpretable model on top of the tracking in [`ice_tracking.md`](ice_tracking.md).

Code: [`scripts/drift_model.py`](../scripts/drift_model.py) (the model),
[`scripts/build_drift_training.py`](../scripts/build_drift_training.py) (training data),
[`scripts/train_drift_model.py`](../scripts/train_drift_model.py) (fit and test),
[`scripts/forecast_chukchi.py`](../scripts/forecast_chukchi.py) (forecast and NISAR check),
[`scripts/forecast_osisaf_interp.py`](../scripts/forecast_osisaf_interp.py) (forecast map started from OSI SAF),
[`scripts/wind.py`](../scripts/wind.py) (ERA5 wind). Tests: [`tests/test_drift_model.py`](../tests/test_drift_model.py).

## The model

For one piece of ice and one time step k (about a day):

```
V[k] = a·V[k-1]  +  b·W[k]  +  c·W[k-1]  +  d
```

| Symbol | Meaning |
|---|---|
| `V[k-1]` | ice drift during the previous step (km/day) — the "previous movement" |
| `W[k-1]` | mean 10 m wind during the previous step (m/s) |
| `W[k]` | mean 10 m wind during the step being predicted (m/s) — in real use, a wind **forecast** |
| `a, b, c, d` | coefficients, fitted by least squares |

All vectors are **complex numbers** (x + i·y on the polar-stereographic map). Multiplying by a complex
coefficient both **scales and turns** a vector, which is exactly the physics of wind-driven ice: sea ice
moves at about 2 % of the wind speed, turned 20-40° to the right of the wind (Northern Hemisphere,
Coriolis). `|b|` is that percentage and `-angle(b)` the turning angle.

For several days ahead the model runs step by step: the predicted `V[k]` becomes the input `V[k-1]` of the
next step, and each new arrow starts where the previous one ended.

**Map directions.** EPSG:3413 is centred on the 45° W meridian, so in the Chukchi Sea (≈ 165° W) map "down
and to the right" points towards the North Pole. Wind (given as east/north) is rotated into the map frame
before it is used (`east_north_to_map()`).

## Training data

NISAR gives one drift field per pair of passes, far too few to learn from. So the model is fitted on a
long record of the same quantities:

| | |
|---|---|
| Ice drift | **OSI SAF low-resolution sea-ice drift** (OSI-405, multi-sensor), 62.5 km grid, one 48 h displacement per day. Only "nominal quality" cells |
| Wind | **ERA5** reanalysis, hourly 10 m wind (via the Open-Meteo archive API), averaged over the same 48 h |
| Area | Chukchi + western Beaufort Sea, 70-80° N, 150-180° W (250 grid cells) |
| Period | 15 Sep - 30 Nov, 2018-2026, windows every second day |
| Samples | 30,769 (cell × pair of back-to-back 48 h windows) |

## How well it works on data it never saw

Fitted on 2018-2023, scored on 2024-2026 (11,875 samples, mean drift 10.5 km/day):

| Method | Error (RMSE, km/day) | vs persistence |
|---|---|---|
| zero (no motion) | 12.0 | 2 % better |
| persistence (tomorrow = today) | 12.3 | — |
| wind only (`b·W + d`) | 5.2 | 58 % better |
| **full model** | **5.0** | **59 % better** |

The model used for forecasts is refitted on 2018-2025. Its coefficients:

| Term | Value | Meaning |
|---|---|---|
| wind now (`b`) | 2.16 % of the wind, 21° to the right | textbook free drift |
| previous drift (`a`) | 0.27 × the previous drift | some drift persists (mostly ocean currents) |
| wind before (`c`) | 0.66 % of the previous wind, pointing backwards | removes the part of the previous drift that was caused by the previous wind |
| constant (`d`) | 1.0 km/day | mean drift in this area |

Persistence is worse than predicting no motion: the wind changes direction from one window to the next,
and the ice follows within hours. The wind of the step being predicted carries almost all the skill.

## Check against NISAR

`forecast_chukchi.py` starts from the 27 → 28 Oct NISAR arrows (176) and predicts the next two days, then
tracks two more NISAR pairs with the same method to see what the ice really did:

| Pair | Gap | NISAR arrows |
|---|---|---|
| 27 Oct 11:01 → 28 Oct 10:20 | 23.3 h | 176 (input) |
| 28 Oct 10:20 → 29 Oct 09:39 | 23.3 h | 128 (truth, day 1) |
| 29 Oct 09:39 → 30 Oct 10:37 | 25.0 h | 11 (truth, day 2) |

A forecast arrow is compared with the NISAR arrow that starts within 4 km of it. Errors are for the
displacement over the day:

| Test | Arrows compared | Model | Wind only | Persistence | Observed mean displacement |
|---|---|---|---|---|---|
| Day 1, 1-day forecast | 64 | **1.6 km** | 1.3 km | 12.3 km | 10.7 km |
| Day 2, 2-day forecast (from day 0 only) | 3 | **2.3 km** | 2.3 km | 25.1 km | 20.9 km |

On day 1 the ice turned by about 70° compared with day 0 (persistence: 67° direction error). The model,
driven by the wind, got the direction within 6° and the length within 0.6 km.

`figures/chukchi_drift_forecast_check.png`: the comparison piece of ice by piece of ice. For every forecast
arrow with a NISAR arrow starting within 4 km: the start (white), where NISAR saw the ice go (blue), where the
model said it would go (orange), the error between the two (red) and persistence (dashed).

`figures/chukchi_drift_forecast.png`: all 176 forecast arrows (orange) and all NISAR-tracked arrows (blue)
at map scale, day 1 (left) and day 2 (right), same map extent in both panels. NISAR arrows exist only where
two passes overlap and the tracking succeeded: 128 on day 1, but only 11 on day 2 (the 29/30 Oct passes
overlap partly and the ice moved ≈ 20 km, near the largest shift the tracking finds). Scores use only the
forecast arrows that start within 4 km of a NISAR arrow (64 and 3).

## 7-day forecast

The same step-by-step forecast, run for 7 steps of 24 h from 28 Oct 10:20 UTC for all 176 pieces of ice
(each starting at the end of its 27 → 28 Oct NISAR arrow). NISAR does not cover where the ice goes after
two days, so the check is against trajectories through the **OSI SAF drift fields of the same days**
(62.5 km, 48 h windows; 2026 is never used for fitting). Median distance between forecast and OSI SAF
positions:

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| ERA5 wind (m/s) | 5.8 | 8.7 | 4.4 | 3.0 | 1.5 | 2.5 | 4.5 |
| Distance from start, OSI SAF (km) | 5.6 | 11.4 | 17.4 | 20.9 | 24.4 | 19.2 | 13.8 |
| **Model** (km) | **6.5** | **14.4** | **12.9** | **15.2** | **15.6** | **16.9** | **16.8** |
| Wind only (km) | 6.2 | 14.4 | 13.8 | 15.7 | 16.8 | 18.0 | 17.9 |
| Persistence (km) | 8.6 | 20.9 | 33.3 | 46.7 | 60.3 | 69.3 | 78.4 |

**Read with care: OSI SAF is not accurate enough to score the model to a few km.** Its 48 h windows smooth
out day-to-day turns. Against NISAR on the two days NISAR covers, OSI SAF itself is off by 5.4 km on day 1
(it gives 5.6 km of motion where NISAR measured 10.4 km) and 9.9 km on day 2 — larger than the model's
own errors (1.6 and 2.3 km). The model–OSI SAF distance of ≈ 15 km from day 2 on is therefore mostly OSI
SAF error. What the comparison does show: the forecast stays on the same track as the observed drift for
the whole week (the distance stops growing after day 2), while persistence drifts 78 km away.

`figures/chukchi_drift_forecast_7day.png`: 7-day paths of one piece of ice per 25 km cell (31 of 176):
model (orange, a dot per day), OSI SAF (blue dashed) and persistence (grey dotted).

## Forecast map started from OSI SAF (10 km grid)

NISAR covers only where it passes. To forecast the whole area, `forecast_osisaf_interp.py` starts from an
OSI SAF drift map instead:

1. **Starting drift:** the OSI SAF map of 26 Oct 12:00 → 28 Oct 12:00 (ends at the forecast start, so it holds
   nothing about the forecast days), interpolated linearly from its 62.5 km cells to a **10 km grid** over the
   Chukchi / Beaufort box (5,300 points).
2. **Wind:** ERA5, interpolated to the same points.
3. **Forecast:** the trained model, 48 h steps on the fixed grid, each step from the previous prediction.

The model is not retrained for this: it is trained at the resolution the data was measured at (62.5 km) and
applied at any point. Training on interpolated points would add no information, only copies of the same
measurements.

**Checked against NISAR-tracked drift.** Starting from OSI SAF interpolated to the Chukchi NISAR points gives a
forecast as good as starting from the NISAR drift itself (28 → 29 Oct, 64 pieces of ice with a NISAR arrow,
same wind):

| Starting drift | Error vs NISAR-tracked drift, day 1 |
|---|---|
| **OSI SAF, interpolated** | **1.2 km** |
| NISAR | 1.6 km |
| none (wind only) | 1.3 km |

The starting drift matters little because only about a quarter of it carries into the next step; the wind
does the rest. This was a one-off check: the script ran it as "part A" in commit `466bfdc`; it was then removed
to keep the script to the forecast map.

**Checked against OSI SAF**, per 48 h step (vector error, km/day):

| Lead | +2 d | +4 d | +6 d | +8 d | +10 d | +12 d |
|---|---|---|---|---|---|---|
| Model, 10 km grid | 3.4 | 4.3 | 4.2 | 3.7 | 4.3 | 4.0 |
| Model, 62.5 km cells | 3.8 | 4.5 | 3.9 | 3.9 | 4.5 | 4.2 |
| Persistence | 11.8 | 9.8 | 10.5 | 9.5 | 13.3 | 12.5 |

The 10 km and 62.5 km scores are the same: the model is linear and drift and wind are interpolated linearly, so
the 10 km forecast is the 62.5 km forecast interpolated afterwards. It gives a value at every point, not more
detail. `figures/osisaf_interp_grid_forecast.png`: forecast speed (colour) and arrows per step.

## Run it

```bash
python scripts/download_scenes.py         # NISAR scenes, incl. the 29 and 30 Oct passes (Earthdata login)
python scripts/build_drift_training.py    # OSI SAF + ERA5, ~350 small files, a few minutes; cached in data/
python scripts/train_drift_model.py       # fit, held-out test, writes models/drift_model.json
python scripts/forecast_chukchi.py        # NISAR tracking of 3 pairs, 2-day and 7-day forecasts, figures
python scripts/forecast_osisaf_interp.py  # forecast map on a 10 km grid, started from OSI SAF
python -m pytest tests/test_drift_model.py
```

## Limitations

- **Perfect wind.** The forecast uses ERA5 reanalysis wind, i.e. the wind that actually blew. A real
  forecast would use forecast wind (GFS, ECMWF), which adds the wind-forecast error; beyond 3-5 days that
  usually dominates.
- **Trained on 48 h steps, applied to ~24 h steps.** OSI SAF drift is 48 h; the NISAR passes are a day
  apart. The wind response (`b`) carries over directly; the persistence terms (`a`, `c`) may differ for
  24 h.
- **Small NISAR check.** Day 1 is one day and 64 arrows in one area. Day 2 rests on 3 arrows: the ice
  moved ≈ 20 km that day, close to the largest shift the tracking can find, and the 29/30 Oct passes
  overlap only partly.
- **No precise check beyond day 2.** Days 3-7 are compared with OSI SAF only, which is too coarse to
  measure errors below ≈ 10-15 km (see above). More NISAR passes along the forecast paths would fix that.
- **Coarse training data.** OSI SAF cells are 62.5 km; the model cannot learn behaviour at the scale of
  single floes or near coasts, where internal ice stress slows the drift.
- **One region, one season.** Coefficients are for the Chukchi / Beaufort Sea in autumn. Refit for other
  areas or seasons.

## Next steps

- Replace ERA5 by archived **forecast** winds (e.g. GFS from the day before) to measure the real forecast skill.
- Check against **IABP buoy** tracks (exact daily positions).
- More NISAR pairs (several passes per day cover this area in late October) for a larger check.
