# Handoff: where the project stands (written 14 Nov 2026, evening)

NASA Space Apps 2026, challenge "Dancing with the SARs". **Deadline: Sun 15 Nov 2026, 23:59 local.**
Goal: (1) detect sea ice in radar images, (2) track and predict where it moves.

## Status

| Part | Status |
|---|---|
| Ice / open-water detection | **Done and frozen. Do not retune.** |
| Ice tracking between two images | Works (NISAR pair, cross-sensor with Sentinel-1) |
| Trajectory *prediction* | **Not started** |
| App / interface | **Not started** |
| Presentation | **Not started** |
| Step-by-step notebook for NISAR | Not built (original notebook untouched) |
| Branch `nisar-loader` -> `main` | Not merged, no pull request yet |

## Plan for 15 Nov (agreed with the user)

1. Trajectory prediction. Sentinel-1 is used **for tracking only** (decided), not for detection.
2. App / interface.
3. Presentation.

First building blocks for trajectory:
- A buoy loader (IABP). Buoys are the only reference with real motion. Best spot: 73-75 N, 155-162 W, 27-28 Oct.
- Decide on a driver for prediction (wind?). Not looked at yet. Without it, only "continue the last drift" is possible.

## Detection result (final)

- Model: boosted trees on 7 NISAR features (HH, VV, HH minus VV, local mean and spread). File: `models/ice_water.joblib`.
- Training (22 scenes, each held out): 94.3 % balanced accuracy. **Locked test (4 scenes, 29-31 Oct): 89.6 %.**
- Labels: Uni Bremen AMSR2, 3 km (third party). Model input: NISAR only.
- Limits: small test set; coarse labels; calm water and thin new ice can fool it; needs HH+VV (gone from 1 Nov);
  about 19 % of scenes have an empty channel and are skipped.

## Tracking result so far

- NISAR 27 -> 28 Oct (pack ice, about 76 N 165 W): 735 vectors, median 9.2 km/day.
- Via Sentinel-1 (NISAR -> S1 -> NISAR) ends within 50 m of the direct NISAR result.
- One buoy inside the area: 11.85 km/day at 38 deg; NISAR 11.44 km/day at 37 deg.
- Sentinel-1-only pairs are weak (38 vectors in 24 h; 48 h fails).

## Where things are

| What | Where |
|---|---|
| Results, limits, how to reproduce | `docs/DETECTION.md` |
| Exact scenes, area, dates | `docs/DATASET.md`, `config/dataset.json` |
| Reference data and drawbacks | `docs/REFERENCE_DATA.md` |
| Which data is used where, third parties | `docs/DATA_SOURCES.md` |
| Why we chose this data | `docs/DATA_CHOICES.md` |
| Code | `scripts/` (48 tests in `tests/`, run `.venv/bin/python -m pytest -q tests`) |
| Tracking run | `scripts/run_chukchi.py` (writes MongoDB + `figures/chukchi_*.png`) |
| Raw data (not in git) | `data/`, 23 GB. 8.7 GB of it is ICESat-2 in `data/reference/icesat2/`, safe to delete |

## Practical notes

- Python env: `.venv` in the repo. Packages: `requirements-nisar.txt`.
- NASA Earthdata login: `~/.netrc` of `claude_user`. Use Python `requests`/`asf_search`; `curl -n` gets a 401 at NSIDC.
- MongoDB: connection string in git-ignored `.env` (`MONGODB_URI`). Database `NasaData`:
  `scenes` (5), `results` (per-scene scores + test result), `ice_motion` (1,243 points). Never print the string.
- Commits use the GitHub no-reply address (the repo is public). Work on a branch, never push to `main`.
- Discord project channel: images were posted there on 14 Nov.
- Before submitting: check citation wording for Uni Bremen, IABP, Copernicus, NASA DAACs, and the rule on declaring AI use.
