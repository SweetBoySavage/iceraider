# Route interface: can the ship pass?

Shows a ship route through the Chukchi Sea on our NISAR ice map and says whether the route is safe.

## Start it

```bash
pip install numpy pyproj          # already there if you installed requirements-nisar.txt
python scripts/run_app.py         # opens http://127.0.0.1:8000/   (options: --port 8001, --no-browser)
```

- Nothing else is needed. The ready-made maps are in `app/data/` (in git, about 13 MB).
- The page shows a loading bar and unlocks when everything is loaded.

## What you see

| Thing | Meaning |
|---|---|
| White / blue | Sea ice / open water, from the frozen detection model |
| Dark | No radar image there |
| Red-orange band | Safety margin around the ice |
| Yellow band | Safety margin around the coast |
| Route colour | Green = observed open water, grey dashes = no radar image, red = inside ice margin, yellow = inside coast margin |
| Dashed ring around the ship | The ice clearance, to scale |
| Labels on the map | Time (UTC) of the radar image used there |

## Controls

- **Scenario:** switch between the two periods. Each has its own map and route. Both are loaded at the start.
- **Ship length** (m) and **speed** (knots).
- **Margin around ice** and **around coast** (default 1,000 m each).
- **Time:** slider, Play, or the arrow keys (left / right = one step back / forward; step is 1, 3, 6 or 12 h). Space = play.
- **Route:** drag a route point. Scroll = zoom, drag the map = move.

## The verdict

| Verdict | When |
|---|---|
| NOT SAFE | Ice or coast comes inside the margin somewhere. The first place and time are shown. |
| CLEAR WHERE OBSERVED | No conflict, but part of the route has no radar image. |
| SAFE | The whole route is observed open water outside all margins. |
| NO DATA | No radar image covers the route. |

Rules (all in `scripts/route_safety.py`):
- The ship is a circle; its diameter is the ship length; its centre is the position.
- Ice conflict: observed ice closer than half the ship + the ice margin.
- Coast conflict: land closer than half the ship + the coast margin.
- No radar image = not verified, never "safe".

## Data: two scenarios (`config/scenarios.json`)

| | 27 - 31 Oct 2026 | 15 - 21 Sep 2026 |
|---|---|---|
| NISAR scenes used | 9 (6 more skipped: dead channel) | 8 |
| Scene kind | training, test, one coastal scene | **training only** |
| Route | 1,250 km along the ice edge | 1,000 km offshore at 75 N |
| Route with a radar image | about 70 % | about 96 % |
| Same ice seen more than once | two pairs, 18 - 23 h apart | three images in two days (15, 16, 17 Sep) |

- Where two scenes overlap, the newest is shown.
- Cells of 400 m. A cell is ice when at least 25 % of it is ice.
- September: the model was trained on these images, so its ice map looks better there than on new images.
- The time bar shows how old the radar image under the ship is.

## Limits (say these in the presentation)

- **The ice does not move yet.** Every image is shown as it was taken; images are up to 4 days apart.
  The trajectory prediction will later give one ice map per time step.
- To add a scenario: add it to `config/scenarios.json`, then rebuild the map (below).
- The margin is fixed. Later it can grow with ship speed, ice drift and the age of the image.
- Distances are measured on the map: up to about 1 % off at 74 N.
- Ice / water close to the coast is not validated. The strip next to the coast counts as "not verified".

## Rebuild the map (only when the data or model changes)

```bash
python scripts/build_mosaic.py 20261027 20261031     # about 5 min each, needs the NISAR files in data/
python scripts/build_mosaic.py 20260915 20260921
python scripts/build_app_data.py                     # writes app/data/<scenario>/
```

## Files

| File | Job |
|---|---|
| `scripts/build_mosaic.py` | Stitch the scene maps into one map |
| `scripts/build_app_data.py` | Make the files the app loads |
| `scripts/route_safety.py` | The safety rules |
| `scripts/run_app.py` | Small web server: page + `/api/check` |
| `app/` | The page (plain HTML, CSS, JavaScript; no libraries) |
