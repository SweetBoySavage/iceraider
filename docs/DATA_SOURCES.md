# Data sources: what was used where

Short answer:
- **The ice / open-water model uses NISAR only.** Sentinel-1 is not an input to it.
- **Sentinel-1 is used only for ice movement** and as a second radar view.
- **Five third-party (non-NASA) sources are used:** the Uni Bremen ice map (labels), the IABP buoys (drift check),
  the GSHHG coastline (land mask), and OSI SAF ice drift + ERA5 wind (drift forecast).

## Which data is used in which step

| Step | NISAR | Sentinel-1 | Other data | Script |
|---|---|---|---|---|
| Ice / open-water **model input** | **yes** (HH + VV) | no | none | `detection.py`, `train_detection.py` |
| **Labels** to train and test the model | no | no | Uni Bremen AMSR2 ice map (third party) | `reference.py` |
| Extra ice confirmation | no | no | ICESat-2 ATL07QL (NASA). Gave no usable labels | `reference.py` |
| Choosing the training / test scenes | scene footprints | no | Uni Bremen AMSR2 ice map | `select_dataset.py` |
| Ice **movement**, main result | **yes** (27 -> 28 Oct) | no | none | `ice_tracking.py`, `run_chukchi.py` |
| Ice movement, cross-sensor | yes | **yes** (26, 27 Oct) | none | `run_chukchi.py`, `s1_geocode.py` |
| Check of ice movement | yes | no | IABP buoys (third party), one-off check, no script in repo | - |
| Early brightness classes (k-means), before the model | yes | yes (27 Oct) | none | `run_chukchi.py` |
| **Land mask** on output maps | no | no | GSHHG coastline (third party) | `land_mask.py`, `evaluate_detection.classify_scene` |
| **Drift forecast**, training and held-out test | no | no | OSI SAF ice drift; ERA5 wind via Open-Meteo (third party) | `build_drift_training.py`, `train_drift_model.py` |
| **Drift forecast** for the NISAR arrows, and its check | **yes** (27, 28, 29, 30 Oct) | no | ERA5 wind via Open-Meteo | `forecast_chukchi.py` |
| Overview and verification maps | yes | no | Uni Bremen AMSR2; NASA NISE_A2 (printed, not used for scores) | `verify_chukchi.py` |

## All sources

### NASA data

| Data | Provider | What we took | Access |
|---|---|---|---|
| **NISAR L2 GCOV** (L-band radar, provisional) | NASA / ISRO mission, via NASA ASF DAAC | 34 scenes for the model, plus 4 for tracking and verification. HH + VV, 80 m | Earthdata login |
| **ICESat-2 ATL07QL** (laser, sea ice) | NASA, via NSIDC DAAC | Passes within 6 h of a scene. No labels came out of it | Earthdata login |
| **NISE_A2** (ice and snow, 25 km) | NASA NSIDC DAAC. The sensor (AMSR2) is Japanese (JAXA) | 29 Oct, as a NASA cross-check only | Earthdata login |
| NASA **CMR** and **ASF** search | NASA | Finding scenes and files | open |

### Data from other agencies, downloaded from NASA

| Data | Provider | What we took | Access |
|---|---|---|---|
| **Sentinel-1 EW GRD** (C-band radar, HH + HV) | ESA / EU Copernicus programme. We downloaded it from NASA ASF DAAC | 4 scenes (26, 27, 28, 29 Oct). 28 Oct did not cover the area; 29 Oct was not used in the final tracking | Earthdata login |

Sentinel-1 is not NASA data. The usual credit line is: *"Contains modified Copernicus Sentinel data 2026"*.

### Third-party data (not NASA, not from a NASA server)

| Data | Provider | Role | Access |
|---|---|---|---|
| **AMSR2 ASI sea ice concentration**, 3.125 km, v5.4 | University of Bremen, Institute of Environmental Physics. Sensor: AMSR2 on GCOM-W (JAXA) | **The only source of labels** for training and testing. Also used to pick scenes and draw maps | open, `data.seaice.uni-bremen.de` |
| **IABP drifting buoy positions** | International Arctic Buoy Programme, hosted by the University of Washington (APL) | One comparison with our NISAR ice drift (27 - 28 Oct) | open, `iabp.apl.uw.edu` |
| **GSHHG 2.3.7 coastline**, full resolution (`gshhs_f.b`) | Wessel and Smith; distributed by the University of Hawaii (SOEST) and NOAA. Licence: LGPL | Land mask applied to the model output. Not used for training or scoring | open, `soest.hawaii.edu/pwessel/gshhg` |
| **OSI SAF low-resolution sea-ice drift** (OSI-405, merged), 62.5 km, 48 h | EUMETSAT OSI SAF, produced by MET Norway. Licence: CC BY 4.0 | Training and held-out test of the drift forecast model (autumns 2018-2026, Chukchi / Beaufort) | open, `thredds.met.no` |
| **ERA5 reanalysis**, hourly 10 m wind | ECMWF / Copernicus Climate Change Service, served by the Open-Meteo archive API | Wind input of the drift forecast model | open, `archive-api.open-meteo.com` |

Before presenting, look up the exact citation each provider asks for (Bremen, IABP, NASA DAACs).

### Third-party code

| Code | From | Role |
|---|---|---|
| Foundation repository `ichdamola/arctic-sea-ice-sar` | GitHub | Starting point. Its notebook (`notebooks/sea_ice_sar.ipynb`) is unchanged. We took over its Lee filter, Otsu and k-means methods into `scripts/classify.py` |
| `scripts/asf_loader.py` | written by the team, then fixed | Makes an ASF Sentinel-1 download usable by the notebook |
| Open-source Python packages | see `requirements-nisar.txt` | numpy, scipy, scikit-learn, scikit-image, h5py, rasterio, pyproj, shapely, asf_search, pymongo, matplotlib |

## In the repository but NOT used by us

- The original notebook's data: one Sentinel-1 scene from 2 May 2024 from **Microsoft Planetary Computer**, with a 2024 Uni Bremen map.
- `scripts/download_ai4arctic.py` and `research/PLAN.md` (AI4Arctic dataset, DTU): part of the foundation repo, never run.
- Sentinel-1C IW RAW (`.dat`) files: Level-0, not usable, ignored.

## Not available, so not used

NASA AU_SI12 (ends 31 Aug 2025), NSIDC-0748 and NSIDC-0116 ice motion, optical images (NASA HLS). See `REFERENCE_DATA.md`.

## Use of AI

Most of the code and documentation in `scripts/`, `tests/` and `docs/` was written with an AI assistant (Claude Code)
and directed and reviewed by the team. Check the Space Apps rules on declaring AI use and state it in the submission.
