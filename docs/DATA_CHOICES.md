# Data choices and why we made them

Use this to defend the scope in the presentation. All numbers were measured on 14 Nov 2026.

## Decisions

| Choice | Decision | Main reason |
|---|---|---|
| Sensor for ice detection | **NISAR only** (L-band, HH + VV) | NASA data is central; HH minus VV separates water from ice |
| Sensor for ice movement | NISAR, **plus Sentinel-1 for extra time points** | Cross-sensor tracking agrees within 50 m |
| Sentinel-1 for detection | **No** | Needs its own model; no time before the deadline |
| Area | Chukchi + western Beaufort Sea, 70-78 N, 150-180 W | Only the Pacific side has NISAR HH+VV; possible shipping route; buoys present |
| Period | 14 Sep - 31 Oct 2026 | HH+VV stops around 1 Nov; before mid-Sep it is mostly open water |
| Labels | Uni Bremen AMSR2, 3 km | The only daily ice map available for these dates |

## Why NISAR and not Sentinel-1 for detection

Daily share of the area that each sensor sees (average):

| Period | NISAR HH+VV | Sentinel-1 EW |
|---|---|---|
| 15 - 31 Oct 2026 | 73 % (about 59 % after the quality gate) | 43 % |
| 1 - 12 Nov 2026 | 0 % | 56 % |

- In October NISAR covers more, and evenly (62 - 79 % every day). Sentinel-1 swings between 0 % and 91 %.
- Our model is trained on L-band HH + VV. Sentinel-1 is C-band HH + HV. The model cannot read it.
- A Sentinel-1 model would need: thermal noise removal in HV, viewing-angle correction, new training tables and a
  new locked test. Estimate 4 - 6 hours, with an uncertain result.
- Sentinel-1 ice detection has a known weak spot: windy open water looks like ice in HH.
- Caveat on November: NISAR files also thin out after 3 Nov, which may partly be processing delay at NASA.

## Why not all NISAR scenes can be used

- The model needs both HH and VV. In about **19 %** of scenes (76 of 392 checked, 14 Sep - 31 Oct) one channel is
  empty: values near -80 dB. It is always the secondary frequency band ("B").
- Per week the share is 10 - 26 %. On 29 Oct it was 78 % (21 of 27), which is why 6 of 10 test scenes were lost.
- NASA's quality summary marks these files "PASS". Our loader catches them (`DeadChannelError`).
- The small `QA_STATS.h5` file next to each scene shows it before download (mean value below 1e-6). We chose not to
  build that check.

## What if only one channel is available

Scored on training scenes only, each held out in turn:

| Input | Balanced accuracy |
|---|---|
| HH + VV (frozen model) | 94.3 % |
| HH only | 86.8 % |
| VV only (brightness alone) | 67.1 % |

From 1 Nov NISAR records VV only in this area, so the model cannot run there. This is a stated limit.

## Why this area

- NISAR HH+VV exists only on the Pacific side (Chukchi, Beaufort, Amundsen Gulf). Near Greenland and Baffin Bay it is
  single channel. HH+HV scenes exist only over land.
- The Chukchi Sea is a possible shipping route, which is the use case.
- About 39 drifting buoys were in the sector on 27 - 28 Oct, about 12 of them at 73 - 75 N, 155 - 162 W.

## Why not finer reference data

- Optical images (30 m, NASA HLS) only cover coasts; in Sep - Oct there is no ice there, and it is dark from late Oct.
- ICESat-2 has no data over open ocean and gave no usable labels.
- NASA AU_SI12 ends 31 Aug 2025. NASA NISE_A2 is empty on 24 - 28 Oct.
- So training and test labels are 3 km. Our map is 80 m. See `REFERENCE_DATA.md`.

## How to say it in one sentence

"For October 2026 in the Chukchi and Beaufort Seas, NISAR alone separates sea ice from open water on about 59 % of the
area per day, with 89.6 % agreement on unseen scenes; Sentinel-1 adds time points for ice movement; a Sentinel-1
detection model for November is the next step."
