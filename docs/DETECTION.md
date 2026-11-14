# Sea ice vs open water from NISAR: summary

Status: **finished and frozen on 14 Nov 2026.** Do not tune the model on the test set.

## Result in one table

| | Scenes | Balanced accuracy | Water found | Ice found |
|---|---|---|---|---|
| Training (each scene scored while held out) | 22 | 94.3 % | 94.0 % | 94.7 % |
| **Test (locked, run once)** | **4** | **89.6 %** | **95.7 %** | **83.5 %** |
| Simple rule on training scenes, for comparison | 22 | 88.2 % | 83.3 % | 93.2 % |

"Balanced accuracy" = the average of "water found" and "ice found".

## What the model is

- **Input:** one NISAR L2 GCOV scene (L-band, HH and VV, 80 m pixels, provisional products).
- **Output:** per pixel, sea ice or open water.
- **Model:** boosted decision trees (scikit-learn `HistGradientBoostingClassifier`, depth 4, 150 trees).
- **7 features per pixel**, all in dB after a 7x7 Lee speckle filter:
  `hh`, `vv`, `ratio` (HH minus VV), and for HH and ratio a 9x9 local mean (`*_smooth`) and local spread (`*_texture`).
- **Why it works:** open water reflects VV about 5 dB more strongly than HH. For sea ice they are about equal.

## Data

- **Area:** Chukchi Sea and western Beaufort Sea, 70-78 N, 150-180 W. No land in any scene.
- **Training:** 14 Sep - 28 Oct 2026, 24 scenes chosen, 22 used.
- **Test:** 29 - 31 Oct 2026 (strictly later), 10 scenes chosen, 4 used.
- The exact scene names are in `config/dataset.json`.
- **Quality gate:** a scene is skipped when one channel carries no signal (median below -60 dB).
  This removed 2 training and 6 test scenes. NISAR stopped recording HH+VV here around 1 Nov, and many
  scenes of 29-31 Oct already have one empty channel.

## Labels (the "truth")

- Uni Bremen AMSR2 sea ice concentration of the same day, 3.125 km cells.
- **Open water:** the cell and all 8 neighbours are below 15 % ice.
- **Ice:** the cell and all 8 neighbours are at 80 % or more.
- Everything else (the blurry ice edge, land) is "unsure" and never used.
- **ICESat-2** (NASA laser, about 100 m) is wired in to confirm ice, but added nothing here: no clear-sky
  pass fell inside a usable scene. It cannot confirm open water (no data over open ocean).

## Known limits (be honest about these in the presentation)

1. **The test set is small:** 4 scenes, no all-water scene. 89.6 % is an estimate with a wide margin.
2. **The reference is coarse (3 km), our map is 80 m.** In one test scene (31 Oct 08:19) the model finds only
   53 % of the reference ice. The radar image shows dark patches inside that "ice", so part of this may be the
   reference, not the model. We cannot prove which.
3. **Glass-calm water** (20 Sep) sends almost nothing back in both channels, so it can be taken for ice.
4. **Thin new ice** (3 Oct) can look like water.
5. **Only tested in one region and one season** (autumn freeze-up, Pacific side of the Arctic).
6. **No land class.** Scenes with land were left out.

## Reproduce

Setup (once), Python 3.12:

```
python -m venv .venv
.venv/bin/pip install -r requirements-nisar.txt
.venv/bin/python -m pytest -q tests          # all tests should pass
```

**Quick path, no downloads (about 2 minutes).** The sampled pixel tables are in `tables/`.

```
.venv/bin/python scripts/train_detection.py                                   # compare the candidate models
.venv/bin/python scripts/train_detection.py freeze "boosted trees: all but angle"
```

The frozen model is already in `models/ice_water.joblib`, with its test result in `models/test_result.json`.

**Full path, from the satellite files (about 12 GB of NISAR, plus up to 9 GB of ICESat-2).**
Needs a NASA Earthdata login in `~/.netrc`: `machine urs.earthdata.nasa.gov login <user> password <pass>`

```
.venv/bin/python scripts/download_scenes.py --dataset     # NISAR scenes listed in config/dataset.json
.venv/bin/python scripts/detection.py table train         # builds tables/train.npz (about 20 minutes)
.venv/bin/python scripts/detection.py table test
.venv/bin/python scripts/evaluate_detection.py --no-db    # scores + figures/detection_test.png
```

Use the model on any NISAR HH+VV scene:

```python
from scripts.evaluate_detection import load_model, predict_scene
from scripts.nisar_loader import load_gcov

water, usable = predict_scene(load_model(), load_gcov("data/<scene>.h5"))   # water: True = open water
```

## Files

| File | What it does |
|---|---|
| `scripts/nisar_loader.py` | Reads a GCOV file: power per channel, noise removal, viewing angle, quality gate |
| `scripts/reference.py` | AMSR2 and ICESat-2 readers, certain labels, scoring |
| `scripts/select_dataset.py` | Chooses the training and test scenes |
| `scripts/detection.py` | Features and labelled pixel tables |
| `scripts/train_detection.py` | Compares models on training data, freezes one |
| `scripts/evaluate_detection.py` | Runs the frozen model on the test set |
| `scripts/classify.py` | Lee filter, the simple HH minus VV rule |
