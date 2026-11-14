# Reference data: what we used, where it is, and its weak points

"Reference data" = independent measurements we compare our own results with.
Everything here was checked on real files on 14 Nov 2026 for the Chukchi / western Beaufort Sea, Sep - Nov 2026.

## Overview

| # | Dataset | Tells us | Detail | Used for | Status |
|---|---|---|---|---|---|
| 1 | Uni Bremen AMSR2 ice concentration | ice or water, whole area | 3.125 km, daily | **detection labels** | **used** |
| 2 | IABP drifting buoys | ice movement at single points | GPS, every 30-60 min | **trajectory check** | tried once, works |
| 3 | ICESat-2 ATL07 (NASA) | ice confirmed along a line | about 100 m | extra ice check | wired in, added nothing |
| 4 | NASA NISE_A2 | ice or water, whole area | 25 km, daily | NASA-branded check | read, not used for labels |
| 5 | Sentinel-1 EW GRD | second radar view | 40 m | cross-check, tracking | used |
| 6 | NASA AU_SI12, NSIDC-0748, optical (HLS) | - | - | - | **not available for our dates** |

---

## 1. Uni Bremen AMSR2 sea ice concentration (our label source)

**What it is**
- A daily map of "what percent of each cell is covered by sea ice".
- Made from AMSR2, a Japanese sensor on the GCOM-W satellite. It measures the natural microwave glow of the
  surface (passive), so it is independent of radar. It works at night and through most cloud.
- Algorithm: ARTIST Sea Ice (ASI), version 5.4, from the University of Bremen. **Not a NASA product.**

**Where to find it**
- `https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/<year>/<mon>/Arctic3125/asi-AMSR2-n3125-<YYYYMMDD>-v5.4.tif`
  (`<mon>` is `sep`, `oct`, ...). No login needed. About 0.5 MB per day.
- Code: `scripts/reference.py` -> `download(date, "bremen", folder)` and `read_bremen(path)`.

**What the file contains**
- One GeoTIFF, 3584 x 2432 cells of 3.125 km, projection EPSG:3413 (polar stereographic), whole Arctic.
- One number per cell:

| Value | Meaning |
|---|---|
| 0 | open water **or no data** (the file does not tell them apart) |
| 1 - 100 | ice concentration in percent |
| 120 | land |

**How we turn it into labels** (`reference.certain_labels`)
- Open water: the cell and all 8 neighbours are below 15 %.
- Ice: the cell and all 8 neighbours are at 80 % or more.
- Everything else is "unsure" and not used.

**Drawbacks**
- **Coarse.** One cell is 3.125 km; one NISAR pixel is 80 m. A cell covers about 1,500 NISAR pixels. Leads (cracks),
  small floes and the exact ice edge are invisible. It cannot judge a shipping lane.
- **It is a daily mix.** The map blends all satellite passes of that day. Our NISAR scene is one instant. Ice moves
  10 - 20 km per day here, so the edge can be several cells off at the moment of the radar image.
- **Thin new ice is underestimated.** It can show as low concentration or as water. This matters in freeze-up (Oct).
- **Weather can create false ice** over open water (cloud water and vapour affect the 89 GHz channel used).
- **0 means two things** (water, or no data).
- **It is a concentration, not a class.** "50 %" does not say which half of the cell is ice.
- **Summer:** melt ponds on the ice look like water. We avoided summer.

**Use for trajectory:** none directly (no motion in it). Useful as a mask: only track where it shows ice.

---

## 2. IABP drifting buoys (best reference for ice movement)

**What it is**
- The International Arctic Buoy Programme collects GPS positions of buoys placed on the ice or in the ocean.
- This is real, measured drift at single points. It is the closest thing to ground truth for trajectories.

**Where to find it**
- `https://iabp.apl.uw.edu/Data_Products/Daily_Full_Res_Data/Arctic/FR_<YYYYMMDD>.dat`
- No login. One text file per day, about 0.3 MB, available up to about 2 days ago.
- No loader in the repo yet. It was read with a one-off script on 14 Nov.

**What the file contains**
- Text, `;` separated, one row per position fix:

`BuoyID;Year;Hour;Min;DOY;POS_DOY;Lat;Lon;BP;Ts;Ta`

| Column | Meaning |
|---|---|
| BuoyID | buoy number |
| Year, Hour, Min, DOY | time of the report (DOY = day of year, with decimals) |
| POS_DOY | time of the GPS position (use this one for motion) |
| Lat, Lon | position in degrees |
| BP, Ts, Ta | air pressure, surface temperature, air temperature; `-999` = not measured |

- 27 Oct 2026: 195 buoys Arctic-wide, 39 in our sector (69-80 N, 135-180 W), 24 to 274 fixes per buoy per day.

**What we found with it**
- One buoy was inside our tracked NISAR area on 27 - 28 Oct:
  buoy 11.85 km/day towards 38 degrees; NISAR tracking 11.44 km/day towards 37 degrees.
- A group of about 12 buoys sits at 73 - 75 N, 155 - 162 W.

**Drawbacks**
- **Points, not a map.** A handful of buoys per scene at best, often none.
- **The file does not say if a buoy is on ice or in open water.** A buoy in water drifts with currents and waves,
  not with the ice. Check against the ice map before using one.
- **Some "buoys" do not move** (stations on land, for example at 71.32 N, 156.61 W near Utqiagvik). Filter them out.
- **Uneven timing.** Fixes every 30 min for some, every few hours for others. Interpolate to the radar times.
- **Bad fixes happen** (jumps, repeated positions). Remove speeds that are physically impossible.
- **Lon can be 0-360 or -180-180** depending on the buoy. Normalise it.
- No quality flags in this file.

**Use for trajectory:** the main validation source. Also usable to test a *prediction*: predict where a buoy will be
tomorrow, then look it up.

---

## 3. ICESat-2 ATL07 sea ice height (NASA)

**What it is**
- A NASA laser satellite. It measures surface height along 6 thin tracks and classifies each short segment.
- Works in darkness. Blocked by cloud.

**Where to find it**
- Collection `ATL07QL` (quick look, within days) or `ATL07` (final, about 6 weeks later), version 007, at NSIDC.
- Search: NASA CMR, `https://cmr.earthdata.nasa.gov/search/granules.json?short_name=ATL07QL&...`
- Download needs the Earthdata login in `~/.netrc`. About 350 MB per file (one file = one pass over the Arctic).
- Code: `scripts/detection.py` -> `find_icesat2(...)`, `scripts/reference.py` -> `read_icesat2_ice(path)`.

**What the file contains** (HDF5, per beam `gt1l ... gt3r`, group `sea_ice_segments`)

| Field | Meaning |
|---|---|
| `latitude`, `longitude` | centre of the segment |
| `heights/height_segment_type` | 0 cloud, 1 ice or snow surface, 2-5 specular lead, 6-9 dark lead |
| `heights/height_segment_length_seg` | segment length, about 15 - 120 m |
| `heights/height_segment_quality`, `..._fit_quality_flag` | quality (1 = good; fit 1-3 = usable) |
| `stats/cloud_flag_asr`, `stats/layer_flag` | cloud indicators |
| `stats/ice_conc_amsr2` | the AMSR2 concentration NASA used as a mask |

**Our rule for "certainly ice":** type 1, good quality, fit 1-3, cloud flag 0-1, layer flag clear.

**Drawbacks**
- **No data over open ocean.** ATL07 is only made where a coarse map shows more than 15 % ice. So it can confirm
  ice and leads, never open water.
- **A line, not an image.** 6 tracks, each a few tens of metres wide.
- **Rarely at the right time.** Only 11 passes fell within 6 hours of a NISAR HH+VV scene in a month.
- **Cloud removes most segments.** On 21 Oct no clear-sky segment was left inside our scene.
- **Not simultaneous.** Hours apart, so the ice has moved (about 0.5 km per hour).
- Result for us: **zero usable labels** in training and test. It also filled 8.7 GB of disk.

**Use for trajectory:** very limited. Not recommended.

---

## 4. NASA NISE_A2 (near-real-time AMSR2 ice and snow)

- Same sensor as dataset 1, processed by NASA NSIDC. 25 km cells (EASE-Grid, EPSG:3408), daily.
- Where: `https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/OTHR/NISE_A2/1/<YYYY>/<MM>/<DD>/NISE_AMSR2_<YYYYMMDD>.HDFEOS`
  (Earthdata login). 2 MB per day. Old HDF4 format; `reference.read_nise_north` reads it without extra libraries.
- Values: 1-100 ice %, 255 ocean, 0 / 101 / 103 / 252 land, snow or coast.
- **Drawbacks:** 8 times coarser than dataset 1. **No ice data on 24 - 28 Oct 2026** (the files exist but are empty).
  Use it only to say "a NASA product agrees", not for labels.

---

## 5. Sentinel-1 EW GRD (ESA, downloaded from NASA ASF)

- C-band radar, HH + HV, 40 m pixels, 400 km wide. Not a "truth", but a second, independent radar view.
- Where: NASA ASF DAAC (`asf_search`), product `GRD_MD`, beam mode `EW`. About 240 MB per scene.
- Code: `scripts/asf_loader.py`, `scripts/s1_geocode.py`.
- What we found: tracking NISAR -> Sentinel-1 -> NISAR ends within 50 m of the direct NISAR result.
- **Drawbacks:** not at the same time as NISAR (hours apart). Thermal noise in HV is not removed in our code yet.
  Sentinel-1-only tracking was weak here (38 vectors in 24 h). It cannot label ice vs water by itself.

**Use for trajectory:** fills the time gaps between NISAR passes, and works in November when NISAR is VV only.

---

## 6. Looked for, not usable

| Dataset | Why not |
|---|---|
| NASA **AU_SI12** (AMSR2 12.5 km) | last file is 31 Aug 2025 |
| NASA **NSIDC-0748** (weekly ice motion) | listed in the catalogue, but no files found |
| NASA **NSIDC-0116** (daily ice motion) | ends 31 Dec 2025 |
| **Optical 30 m** (NASA HLS: Landsat, Sentinel-2) | only near coasts; no ice there in Sep - Oct; too dark from late Oct |
| NOAA / US National Ice Center ice charts | not checked |
| Wind and current data (for example ERA5) | not checked; likely needed for trajectory *prediction* |

---

## What this means for trajectory prediction

- **Validation:** IABP buoys are the only reference with real motion. Plan the area and dates around where buoys are
  (73 - 75 N, 155 - 162 W had about 12 on 27 - 28 Oct).
- **No gridded ice-motion reference exists for late 2026.** We cannot compare a whole motion map with a NASA product.
- **Ice mask:** use our own detection model (80 m) where NISAR has HH+VV; fall back to the AMSR2 map elsewhere.
- **Open point:** a prediction needs a driver (wind, currents) or it can only continue the last measured drift.
  We have not looked at wind data yet.
