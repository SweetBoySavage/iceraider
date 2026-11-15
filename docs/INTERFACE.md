# Route interface: can the ship pass?

Shows a ship route through the Chukchi Sea on our NISAR ice map and says whether the route is safe.

## Install and start (about 2 minutes)

You need **git** and **Python 3.12 or newer**. No radar data, no accounts, no database.

```bash
git clone https://github.com/SweetBoySavage/iceraider.git
cd iceraider
git checkout ship-route-interface      # until the branch is merged into main

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install numpy pyproj

python scripts/run_app.py              # opens http://127.0.0.1:8000/
```

- Already have the repo? `git pull`, then `git checkout ship-route-interface`.
- Port in use? `python scripts/run_app.py --port 8001`.
- Show it to another computer on the same network: add `--host 0.0.0.0`, then open `http://<your IP>:8000/` there.
- No browser opens? Go to `http://127.0.0.1:8000/` yourself. Use Chrome, Edge or Firefox.
- Stop the app with Ctrl+C in the terminal.
- The ready-made maps are in `app/data/` (in git, about 13 MB), so nothing is computed at start.
- The page shows a loading bar and unlocks when everything is loaded.

## Use it in 30 seconds

The left side is numbered in the order you use it. It always fits on one screen (the text shrinks, it never scrolls).

1. Pick a **period and route**.
2. Read the **verdict**: safe or not, and how the route splits up. A button jumps to the first conflict.
   In the same box, the check boxes switch map items on and off.
3. Change **ship length, speed and margins** with the sliders. The verdict updates at once.
4. Press **Play** or the **arrow keys** to move the ship along the route.
5. **Drag a route point** to steer around the ice.

## What you see

| Thing | Meaning |
|---|---|
| White / blue | Sea ice / open water, from the frozen detection model |
| Dark | No radar image there |
| Red-orange band | Safety margin around the ice |
| Yellow band | Safety margin around the coast |
| Route colour | Green = observed open water, grey dashes = no radar image, red = inside ice margin, yellow = inside coast margin |
| Dashed ring around the ship | The ice clearance, to scale |
| Labels on the map | Date and time (UTC) of the radar image used there |
| Grey land and place names | The wider area (Alaska, Russia, the seas), from a coarse world outline |

## Controls

- **Scenario:** switch between the two periods. Each has its own map and route. Both are loaded at the start.
- **Show on the map:** check boxes for sea ice, route, both safety margins, NISAR image dates, the latitude / longitude grid and place names.
- **Ship length** (m) and **speed** (knots).
- **Margin around ice** and **around coast** (default 1,000 m each).
- **Time:** slider, Play, or the arrow keys (left / right = one step back / forward; step is 1, 3, 6 or 12 h). Space = play.
- **Route:** drag a route point. Scroll = zoom, drag the map = move.

## The verdict

| Verdict | When |
|---|---|
| NOT SAFE: SEA ICE ON THE ROUTE (or: TOO CLOSE TO THE COAST) | Ice or coast comes inside the margin. The box says after how many km, when, and at which position. |
| NO ICE SEEN, BUT PARTLY UNKNOWN | No conflict, but part of the route has no radar image. |
| SAFE: OPEN WATER ALL THE WAY | The whole route is observed open water outside all margins. |
| UNKNOWN: NO RADAR IMAGE | No radar image covers the route. |

The box shows only the verdict, a bar with the share of the route per colour, and a button that jumps to the first conflict.
The time bar shows the date and time at the ship, the hours since departure and the distance sailed.

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

## What is in this version, and what is not

**Added**

| What | Why |
|---|---|
| Stitched ice map per period (`build_mosaic.py`) | One scene is about 240 km wide; a 1,000 km route needs many scenes |
| Two scenarios with a switch | Oct has test data and the checked tracking; Sep has the best covered route |
| Safety check with margins around ice and coast | The core question: can the ship pass? |
| Ship length and speed sliders | Asked for in the assignment |
| Time slider, play, arrow keys | To follow the ship along the route |
| Draggable route points | Little work, and lets you steer around ice |
| "No radar image" as its own state | We do not call water safe that we never saw |
| Maps committed to git | The app must start anywhere with one command |

**Not added**

| What | Why not |
|---|---|
| Moving ice | The trajectory prediction is not built yet. Every image is shown as taken |
| Hourly ice frames | NISAR sees a spot at most 2 - 3 times in a week. Hourly frames must come from the prediction |
| Margin that grows with speed, drift and image age | Agreed to wait for the prediction. The margin is a fixed number for now |
| Sentinel-1 or AMSR2 to fill gaps | Decision: NISAR only |
| The 6 scenes of 29 - 31 Oct with a dead channel | The model needs HH and VV; one of them is empty there |
| Sea depth | Idea only, no data loaded |
| Automatic route finding | Out of scope; the user moves the route by hand |

**Known weak spots**

- **The clean-up of 15 Nov (evening) is not yet looked at in a real browser.** The first version was checked by the user.
  The clean-up was tested with pytest and by running the page script in Node. Expect small visual fixes.
- Both default routes are NOT SAFE (Oct: ice at km 306; Sep: starts inside ice). Fine for a proof of concept.
- September uses training scenes only, so the ice map looks better there than on new images.

## AI use (Space Apps rule)

- The code of this interface was written with an AI assistant (Claude, Anthropic), directed and checked by the team.
- The page says so in three places: a visible mark at the top right of the map (it shows in every screenshot and
  screen recording), a `meta` tag in the page, and a comment at the top of each file in `app/`.
- The ice map itself is not generated content: it is the output of our own trained model on NISAR data.
- The page uses no NASA logo, flag or mission identifier.
- Still to do by the team: declare the AI use in the project submission form.

## Limits (say these in the presentation)

- **The ice does not move yet.** Every image is shown as it was taken; images are up to 4 days apart.
  The trajectory prediction will later give one ice map per time step.
- To add a scenario: add it to `config/scenarios.json`, then rebuild the map (below).
- The margin is fixed. Later it can grow with ship speed, ice drift and the age of the image.
- Distances are measured on the map: up to about 1 % off at 74 N.
- The map is turned so that north is up in the middle of the route. Away from the middle, north leans a little.
- World outline and coast: GSHHG 2.3.7 (Wessel and Smith), LGPL. The outline outside the radar map is coarse (a point every 20 km).
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
