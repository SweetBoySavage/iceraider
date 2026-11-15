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
- The exact scene list, with date, time, position and whether each scene was used, is in [DATASET.md](DATASET.md) (machine-readable: `config/dataset.json`).
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
6. **Land is not learned, it is masked.** Training and test scenes contain no land. Without a mask the model
   calls land "ice" (97.5 % of land pixels in a check scene near Utqiagvik, 27 Oct). So land is painted over the
   model's answer with a coastline mask (see "Land" below). Waters close to the coast are not validated.

## Land

- Land comes from the **GSHHG full-resolution coastline** (third party, see `DATA_SOURCES.md`), not from the model.
- **Land is applied after the prediction.** The model first answers on the untouched radar data, exactly as in
  training. Then land and the coast margin are painted over it. The model input never changes.
- A **safety margin** is added around all land (default 1 km). Lakes count as land.
- **Land and margin are separate classes:** `LAND` is real land, `COAST_MARGIN` is the strip of sea next to it.
  Land just outside the image still casts its margin into it.
- Set the margin with `LAND_MARGIN_M` (metres) in `.env` or as an environment variable; see `.env.example`.
  A negative or non-number value stops the run. Below 400 m you get a warning: the model looks at a 720 m
  window, so answers that close to the coast are computed from pixels that contain land.
- `classify_scene` returns a map with sea ice, open water, land, coast margin and no-data.
- Check on a real coastal scene: `figures/land_mask_check.png`. The coastline lines up with the radar image.
- Same scene (Utqiagvik, 27 Oct) as a full map: `figures/map_20261027T074517.png`. Land is 6.4 % of the
  scene; the margin adds 0.6 % at 1000 m and 2.5 % at 5000 m.
- Not included: water depth. A shallow-water limit for shipping would need a depth chart.
- First mask of a session takes about 30 s (reading the coastline); it is slow for an app unless cached per scene.

One-time download of the coastline (119 MB zip, 96 MB file, not stored in git):

```
.venv/bin/python -c "from scripts.land_mask import download; download()"
```

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

With land (recommended for anything shown to users):

```python
from scripts.evaluate_detection import COAST_MARGIN, LAND, OPEN_WATER, SEA_ICE, classify_scene, load_model
from scripts.nisar_loader import load_gcov

classes = classify_scene(load_model(), load_gcov("data/<scene>.h5"))        # 0 sea ice, 1 open water, 2 land, 3 coast margin, 255 no data
```

Compute a scene once and save it (about a minute). Writes the map to `data/maps/<scene name>.npz` and a
picture to `figures/map_<start time>.png`:

```
.venv/bin/python scripts/evaluate_detection.py map <scene name>
```

An interface then loads the saved map in well under a second, with nothing computed again:

```python
from scripts.evaluate_detection import load_map

m = load_map("<scene name>")     # m["classes"], m["x"], m["y"], m["epsg"], m["margin_m"]
```

The margin is baked into a saved map. After changing `LAND_MARGIN_M`, run the `map` command again.

## Files

| File | What it does |
|---|---|
| `scripts/nisar_loader.py` | Reads a GCOV file: power per channel, noise removal, viewing angle, quality gate |
| `scripts/reference.py` | AMSR2 and ICESat-2 readers, certain labels, scoring |
| `scripts/select_dataset.py` | Chooses the training and test scenes |
| `scripts/detection.py` | Features and labelled pixel tables |
| `scripts/train_detection.py` | Compares models on training data, freezes one |
| `scripts/evaluate_detection.py` | Runs the frozen model on the test set; three-class map of any scene (`map`) |
| `scripts/classify.py` | Lee filter, the simple HH minus VV rule |
| `scripts/land_mask.py` | Land mask from the GSHHG coastline, with a safety margin |
