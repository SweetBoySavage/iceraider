"use strict";
// Route interface: drawing and controls only. Whether a route is safe is decided by the server (scripts/route_safety.py).

const $ = (id) => document.getElementById(id);
const CLASS = { ICE: 0, WATER: 1, LAND: 2, COAST: 3, NO_DATA: 255 };
const STATUS = ["clear", "unverified", "ice", "land"];
const STATUS_COLOR = ["#2ee6a6", "#9aa7b8", "#ff4d4d", "#ffb300"];
const BASE_COLOR = { [CLASS.ICE]: [232, 243, 250], [CLASS.WATER]: [15, 76, 117], [CLASS.LAND]: [58, 62, 70], [CLASS.COAST]: [15, 76, 117], [CLASS.NO_DATA]: [22, 29, 41] };
const PARAMS = ["ship_m", "speed_kn", "ice_buffer_m", "coast_margin_m"];
const KNOT_KMH = 1.852;
const PLAY_HOURS_PER_SECOND = 6;

const state = { scenarios: {}, scenarioId: null, meta: null, cls: null, iceSteps: null, landSteps: null, base: null, overlay: null, params: {}, route: [], result: null,
                hours: 0, playing: false, view: { scale: 1, ox: 0, oy: 0 }, drag: null, dirty: true, checkId: 0 };

// ---------- map projection (EPSG:3413, polar stereographic, true at 70 N, 45 W down) ----------
const PS = (() => {
  const a = 6378137, e = 0.0818191908426, lon0 = -45 * Math.PI / 180, latTs = 70 * Math.PI / 180;
  const tOf = (lat) => Math.tan(Math.PI / 4 - lat / 2) / Math.pow((1 - e * Math.sin(lat)) / (1 + e * Math.sin(lat)), e / 2);
  const k = a * Math.cos(latTs) / Math.sqrt(1 - e * e * Math.sin(latTs) ** 2) / tOf(latTs);
  return {
    forward(lat, lon) {
      const rho = k * tOf(lat * Math.PI / 180), d = lon * Math.PI / 180 - lon0;
      return [rho * Math.sin(d), -rho * Math.cos(d)];
    },
    inverse(x, y) {
      const t = Math.hypot(x, y) / k;
      let lat = Math.PI / 2 - 2 * Math.atan(t);
      for (let i = 0; i < 6; i++) lat = Math.PI / 2 - 2 * Math.atan(t * Math.pow((1 - e * Math.sin(lat)) / (1 + e * Math.sin(lat)), e / 2));
      let lon = (lon0 + Math.atan2(x, -y)) * 180 / Math.PI;
      if (lon < -180) lon += 360;
      if (lon > 180) lon -= 360;
      return [lat * 180 / Math.PI, lon];
    },
  };
})();
const cellOf = (lat, lon) => { const [x, y] = PS.forward(lat, lon), m = state.meta; return [(x - m.x0) / m.cell_m, (m.y0 - y) / m.cell_m]; };
const latLonOf = (col, row) => { const m = state.meta; return PS.inverse(m.x0 + col * m.cell_m, m.y0 - row * m.cell_m); };
const toScreen = (col, row) => [col * state.view.scale + state.view.ox, row * state.view.scale + state.view.oy];
const toCell = (sx, sy) => [(sx - state.view.ox) / state.view.scale, (sy - state.view.oy) / state.view.scale];

// ---------- text ----------
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const two = (n) => String(n).padStart(2, "0");
const fmtTime = (date) => `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]} ${two(date.getUTCHours())}:${two(date.getUTCMinutes())} UTC`;
const fmtKm = (km) => `${Math.round(km).toLocaleString("en-US")} km`;
const fmtHours = (h) => h >= 48 ? `${(h / 24).toFixed(1)} days` : `${h.toFixed(1)} h`;
const fmtLatLon = (lat, lon) => `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"}  ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`;
const fmtParam = { ship_m: (v) => `${v} m`, speed_kn: (v) => `${v} kn`, ice_buffer_m: (v) => `${(v / 1000).toFixed(1)} km`, coast_margin_m: (v) => `${(v / 1000).toFixed(1)} km` };
const timeAt = (hours) => new Date(Date.parse(state.meta.route.departure_utc) + hours * 3600e3);

// ---------- loading ----------
function progress(share, text) { $("loading-bar").style.width = `${Math.round(share * 100)}%`; $("loading-text").textContent = text; }

async function getJson(url) {
  const reply = await fetch(url);
  const body = await reply.json();
  if (!reply.ok) throw new Error(body.error || `${url}: ${reply.status}`);
  return body;
}

async function loadPixels(url, width, height) {
  const reply = await fetch(url);
  if (!reply.ok) throw new Error(`${url}: ${reply.status}`);
  const bitmap = await createImageBitmap(await reply.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  if (bitmap.width !== width || bitmap.height !== height) throw new Error(`${url}: size ${bitmap.width} x ${bitmap.height}, expected ${width} x ${height}`);
  const canvas = new OffscreenCanvas(width, height), ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, width, height).data;
}

async function load() {                         // every scenario is loaded before the page unlocks: switching is instant
  progress(0.03, "Reading the list of scenarios…");
  const list = await getJson("data/scenarios.json");
  for (const [k, item] of list.entries()) {
    const share = (step) => (k + step) / list.length * 0.85 + 0.05, dir = `data/${item.id}/`;
    progress(share(0), `${item.label}: reading the map description…`);
    const meta = await getJson(dir + "meta.json"), n = meta.nx * meta.ny;
    progress(share(0.15), `${item.label}: loading the ice map…`);
    const cells = await loadPixels(dir + "cells.png", meta.nx, meta.ny);
    progress(share(0.45), `${item.label}: loading distances to ice and coast…`);
    const dist = await loadPixels(dir + "distance.png", meta.nx, meta.ny);
    const cls = new Uint8Array(n), iceSteps = new Uint8Array(n), landSteps = new Uint8Array(n);
    for (let i = 0; i < n; i++) { cls[i] = cells[4 * i]; iceSteps[i] = dist[4 * i]; landSteps[i] = dist[4 * i + 1]; }
    progress(share(0.75), `${item.label}: drawing the map…`);
    await new Promise(requestAnimationFrame);
    state.scenarios[item.id] = { meta, cls, iceSteps, landSteps, base: paintBase(meta, cls) };
  }
  $("scenario").innerHTML = list.map((item) => `<option value="${item.id}">${item.label}</option>`).join("");
  progress(0.92, "Checking the route…");
  await useScenario(list[0].id);
  progress(1, "Ready");
  $("loading").classList.add("done");
}

async function useScenario(id) {
  Object.assign(state, state.scenarios[id], { scenarioId: id, result: null, hours: 0 });
  state.overlay = new OffscreenCanvas(state.meta.nx, state.meta.ny);
  setPlaying(false);
  $("scenario").value = id;
  $("route-name").textContent = state.meta.route.name;
  $("scenario-note").textContent = state.meta.note;
  resetValues(); paintOverlay();
  await check();
  resize(); fitRoute();
}

// ---------- map layers ----------
function paintBase({ nx, ny }, cls) {
  const canvas = new OffscreenCanvas(nx, ny), ctx = canvas.getContext("2d");
  const image = ctx.createImageData(nx, ny), px = image.data;
  for (let i = 0; i < nx * ny; i++) {
    const c = BASE_COLOR[cls[i]] || BASE_COLOR[CLASS.NO_DATA];
    px[4 * i] = c[0]; px[4 * i + 1] = c[1]; px[4 * i + 2] = c[2]; px[4 * i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function paintOverlay() {                      // the safety margins around ice and coast, for the current values
  const { nx, ny, png_step_m: step } = state.meta, p = state.params, ctx = state.overlay.getContext("2d");
  const iceMax = (p.ship_m / 2 + p.ice_buffer_m) / step, landMax = (p.ship_m / 2 + p.coast_margin_m) / step;
  const image = ctx.createImageData(nx, ny), px = image.data, { cls, iceSteps, landSteps } = state;
  for (let i = 0; i < nx * ny; i++) {
    if (cls[i] === CLASS.LAND) continue;
    if (landSteps[i] <= landMax) { px[4 * i] = 255; px[4 * i + 1] = 179; px[4 * i + 2] = 0; px[4 * i + 3] = 210; }
    else if (cls[i] !== CLASS.ICE && iceSteps[i] <= iceMax) { px[4 * i] = 255; px[4 * i + 1] = 90; px[4 * i + 2] = 60; px[4 * i + 3] = 190; }
  }
  ctx.putImageData(image, 0, 0);
  state.dirty = true;
}

// ---------- route check (server) ----------
async function check() {
  const id = ++state.checkId, p = state.params;
  const query = `scenario=${state.scenarioId}&` + PARAMS.map((k) => `${k}=${p[k]}`).join("&") + "&route=" + state.route.map(([lat, lon]) => `${lat.toFixed(5)},${lon.toFixed(5)}`).join(";");
  const result = await getJson(`api/check?${query}`);
  if (id !== state.checkId) return;            // a newer question is already on its way
  state.result = result;
  const total = result.verdict.hours;
  $("time").max = total.toFixed(2);
  state.hours = Math.min(state.hours, total);
  showVerdict(); paintStrip(); showTime();
  state.dirty = true;
}

let checkTimer = null;
function checkSoon() { clearTimeout(checkTimer); checkTimer = setTimeout(() => check().catch(showError), 120); }
function showError(error) { $("verdict-state").textContent = "ERROR"; $("verdict-text").textContent = error.message; console.error(error); }

function showVerdict() {
  const v = state.result.verdict, box = $("verdict");
  box.className = `card ${v.state}`;
  $("verdict-state").textContent = { blocked: "NOT SAFE", clear: "SAFE", clear_where_observed: "CLEAR WHERE OBSERVED", no_data: "NO DATA" }[v.state];
  let text = "";
  if (v.first_conflict) {
    const f = v.first_conflict, what = f.kind === "ice" ? "Sea ice inside the safety margin" : "Coast inside the safety margin";
    text = `${what} at km ${Math.round(f.km)}, ${fmtHours(f.hours)} after departure (${fmtTime(timeAt(f.hours))}).`;
  } else if (v.state === "clear_where_observed") text = `No ice on the observed part. ${fmtKm(v.km.unverified)} has no radar image and is not verified.`;
  else if (v.state === "clear") text = "The whole route is observed open water, outside every safety margin.";
  else text = "No radar image covers this route.";
  $("verdict-text").textContent = text;
  $("v-km").textContent = fmtKm(v.total_km); $("v-hours").textContent = fmtHours(v.hours);
  for (const name of STATUS) $(`v-${name}`).textContent = fmtKm(v.km[name]);
  $("share").innerHTML = STATUS.map((name, i) => `<i style="width:${100 * v.km[name] / v.total_km}%;background:${STATUS_COLOR[i]}"></i>`).join("");
  $("jump").hidden = !v.first_conflict;
}

function paintStrip() {                         // status along the route, lined up with the time slider
  const canvas = $("strip"), r = state.result, width = canvas.width = canvas.clientWidth, ctx = canvas.getContext("2d");
  const total = r.km[r.km.length - 1];
  let i = 0;
  for (let x = 0; x < width; x++) {
    const km = (x + 0.5) / width * total;
    while (i < r.km.length - 1 && r.km[i + 1] <= km) i++;
    ctx.fillStyle = STATUS_COLOR[r.status[i]];
    ctx.fillRect(x, 0, 1, canvas.height);
  }
}

// ---------- time ----------
function shipAt(hours) {                        // position, heading and status of the ship after `hours`
  const r = state.result, km = Math.min(hours * state.params.speed_kn * KNOT_KMH, r.km[r.km.length - 1]);
  let lo = 0, hi = r.km.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (r.km[mid] <= km) lo = mid; else hi = mid; }
  const f = r.km[hi] > r.km[lo] ? (km - r.km[lo]) / (r.km[hi] - r.km[lo]) : 0;
  return { km, col: r.col[lo] + f * (r.col[hi] - r.col[lo]), row: r.row[lo] + f * (r.row[hi] - r.row[lo]),
           heading: Math.atan2(r.row[hi] - r.row[lo], r.col[hi] - r.col[lo]), status: r.status[lo], scene: r.scene[lo] };
}

function setHours(hours) {
  if (!state.result) return;
  state.hours = Math.max(0, Math.min(hours, state.result.verdict.hours));
  showTime(); state.dirty = true;
}

function showTime() {
  const r = state.result; if (!r) return;
  const ship = shipAt(state.hours), scene = state.meta.scenes[ship.scene];
  $("time").value = state.hours;
  $("clock").textContent = `T + ${state.hours.toFixed(1)} h  ·  ${fmtTime(timeAt(state.hours))}`;
  const where = ["open water, observed", "no radar image here", "inside the ice margin", "inside the coast margin"][ship.status];
  $("shipinfo").innerHTML = `km ${Math.round(ship.km)} of ${Math.round(r.verdict.total_km)} · <b style="color:${STATUS_COLOR[ship.status]}">${where}</b>`
    + (scene ? ` · radar image of ${fmtTime(new Date(scene.time_utc))} (${imageAge(scene)})` : "");
}

function imageAge(scene) {                      // how old the radar image is when the ship is there
  const hours = (timeAt(state.hours) - Date.parse(scene.time_utc)) / 3600e3;
  return hours >= 0 ? `${fmtHours(hours)} old` : `taken ${fmtHours(-hours)} later`;
}

function setPlaying(on) { state.playing = on; state.lastFrame = null; $("play").textContent = on ? "❚❚ Pause" : "▶ Play"; state.dirty = true; }
const stepHours = () => Number($("step").value);

// ---------- drawing ----------
const canvas = $("map"), ctx = canvas.getContext("2d");
let dpr = 1;
function resize() {
  dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr); canvas.height = Math.round(canvas.clientHeight * dpr);
  if (state.result) paintStrip();
  state.dirty = true;
}

function fitRoute() {
  const pts = state.route.map(([lat, lon]) => cellOf(lat, lon));
  const cols = pts.map((p) => p[0]), rows = pts.map((p) => p[1]);
  const w = canvas.clientWidth, h = canvas.clientHeight - 110, pad = 70;          // 110: room for the time bar
  const spanC = Math.max(...cols) - Math.min(...cols), spanR = Math.max(...rows) - Math.min(...rows);
  const scale = Math.min((w - 2 * pad) / spanC, (h - 2 * pad) / spanR);
  state.view = { scale, ox: w / 2 - scale * (Math.min(...cols) + spanC / 2), oy: h / 2 - scale * (Math.min(...rows) + spanR / 2) };
  state.dirty = true;
}

function drawGraticule() {
  ctx.lineWidth = 1; ctx.strokeStyle = "#ffffff1c"; ctx.fillStyle = "#8fa3ba"; ctx.font = "11px system-ui";
  const line = (points) => { ctx.beginPath(); points.forEach(([lat, lon], i) => { const [x, y] = toScreen(...cellOf(lat, lon)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); };
  for (let lat = 64; lat <= 80; lat += 2) {
    line(Array.from({ length: 61 }, (_, i) => [lat, -205 + i]));
    for (const lon of [-140.5, -170.5]) { const [x, y] = toScreen(...cellOf(lat, lon)); ctx.fillText(`${lat}° N`, x + 4, y - 3); }
  }
  for (let lon = -200; lon <= -140; lon += 10) {
    line([[62, lon], [82, lon]]);
    for (const lat of [68.1, 74.1]) { const [x, y] = toScreen(...cellOf(lat, lon)); ctx.fillText(lon < -180 ? `${360 + lon}° E` : `${-lon}° W`, x + 4, y - 3); }
  }
}

function drawSceneTimes() {
  if (state.view.scale < 0.12) return;
  ctx.font = "600 11px system-ui"; ctx.textAlign = "center";
  for (const scene of state.meta.scenes) {
    if (scene.col === null) continue;
    const [x, y] = toScreen(scene.col, scene.row), text = fmtTime(new Date(scene.time_utc)), w = ctx.measureText(text).width + 12;
    ctx.fillStyle = "#0a1018b8"; ctx.beginPath(); ctx.roundRect(x - w / 2, y - 10, w, 19, 6); ctx.fill();
    ctx.fillStyle = "#cfe3f5"; ctx.fillText(text, x, y + 4);
  }
  ctx.textAlign = "left";
}

function drawRoute() {
  const r = state.result;
  const path = (from, to) => { ctx.beginPath(); for (let i = from; i <= to; i++) { const [x, y] = toScreen(r.col[i], r.row[i]); i === from ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } };
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.strokeStyle = "#05080dcc"; ctx.lineWidth = 7; path(0, r.col.length - 1); ctx.stroke();
  let start = 0;
  for (let i = 1; i <= r.status.length; i++) {
    if (i < r.status.length && r.status[i] === r.status[start]) continue;
    ctx.strokeStyle = STATUS_COLOR[r.status[start]]; ctx.lineWidth = r.status[start] >= 2 ? 4.5 : 3.5;
    ctx.setLineDash(r.status[start] === 1 ? [3, 7] : []);
    path(start, Math.min(i, r.status.length - 1)); ctx.stroke();
    start = i;
  }
  ctx.setLineDash([]);
  if (state.drag && state.drag.point !== undefined) {                   // the route being dragged, before the new check
    ctx.strokeStyle = "#ffffffaa"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]); ctx.beginPath();
    state.route.forEach(([lat, lon], i) => { const [x, y] = toScreen(...cellOf(lat, lon)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.setLineDash([]);
  }
  state.route.forEach(([lat, lon], i) => {
    const [x, y] = toScreen(...cellOf(lat, lon)), end = i === 0 || i === state.route.length - 1;
    ctx.beginPath(); ctx.arc(x, y, end ? 6.5 : 4.5, 0, 2 * Math.PI);
    ctx.fillStyle = end ? "#4cc2ff" : "#e6eef7"; ctx.fill(); ctx.strokeStyle = "#05080d"; ctx.lineWidth = 1.5; ctx.stroke();
  });
  ctx.font = "600 12px system-ui"; ctx.fillStyle = "#e6eef7";
  const label = (i, text) => { const [x, y] = toScreen(...cellOf(...state.route[i])); ctx.fillText(text, x + 11, y + 4); };
  label(0, "Start"); label(state.route.length - 1, "End");
}

function drawConflict(now) {
  const f = state.result.verdict.first_conflict; if (!f) return;
  const r = state.result, i = r.km.findIndex((km) => km >= f.km - 1e-6), [x, y] = toScreen(r.col[i], r.row[i]);
  const pulse = 10 + 4 * Math.sin(now / 300);
  ctx.strokeStyle = STATUS_COLOR[f.kind === "ice" ? 2 : 3]; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.arc(x, y, pulse, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
}

function drawShip() {
  const ship = shipAt(state.hours), [x, y] = toScreen(ship.col, ship.row), p = state.params;
  const ring = (p.ship_m / 2 + p.ice_buffer_m) / state.meta.cell_m * state.view.scale;      // the ice clearance, to scale
  if (ring > 3) { ctx.strokeStyle = "#ffffff99"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.arc(x, y, ring, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]); }
  ctx.save(); ctx.translate(x, y); ctx.rotate(ship.heading);
  ctx.beginPath(); ctx.moveTo(15, 0); ctx.lineTo(-9, 8); ctx.lineTo(-5, 0); ctx.lineTo(-9, -8); ctx.closePath();
  ctx.fillStyle = "#ffffff"; ctx.strokeStyle = STATUS_COLOR[ship.status]; ctx.lineWidth = 3; ctx.shadowColor = "#000"; ctx.shadowBlur = 8;
  ctx.fill(); ctx.shadowBlur = 0; ctx.stroke(); ctx.restore();
}

function drawScaleBar() {
  const kmPerPx = state.meta.cell_m / 1000 / state.view.scale;
  const km = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((n) => n / kmPerPx >= 70) || 1000, w = km / kmPerPx;
  const x = 18, y = canvas.clientHeight - 128;
  ctx.strokeStyle = ctx.fillStyle = "#e6eef7"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y - 5); ctx.stroke();
  ctx.font = "12px system-ui"; ctx.fillText(`${km} km`, x + 4, y - 7);
}

function draw(now) {
  const { scale, ox, oy } = state.view;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0a1018"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
  ctx.imageSmoothingEnabled = scale < 1;
  ctx.drawImage(state.base, 0, 0); ctx.drawImage(state.overlay, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawGraticule(); drawSceneTimes(); drawRoute(); drawConflict(now); drawShip(); drawScaleBar();
}

function frame(now) {
  if (state.result) {
    if (state.playing) {
      if (state.lastFrame) setHours(state.hours + (now - state.lastFrame) / 1000 * PLAY_HOURS_PER_SECOND);
      state.lastFrame = now;
      if (state.hours >= state.result.verdict.hours) setPlaying(false);
    }
    if (state.dirty || state.playing || state.result.verdict.first_conflict) { state.dirty = false; draw(now); }
  }
  requestAnimationFrame(frame);
}

// ---------- controls ----------
function resetValues() {
  const m = state.meta;
  state.route = m.route.points_lat_lon.map((p) => [...p]);
  for (const key of PARAMS) { state.params[key] = m.defaults[key]; $(key).value = m.defaults[key]; $(`o-${key}`).textContent = fmtParam[key](m.defaults[key]); }
}

function wire() {
  for (const key of PARAMS) $(key).addEventListener("input", () => {
    state.params[key] = Number($(key).value); $(`o-${key}`).textContent = fmtParam[key](state.params[key]);
    if (key !== "speed_kn") paintOverlay();
    checkSoon();
  });
  $("scenario").addEventListener("change", () => useScenario($("scenario").value).catch(showError));
  $("reset").onclick = () => { resetValues(); paintOverlay(); setHours(0); fitRoute(); checkSoon(); };
  $("jump").onclick = () => { setPlaying(false); setHours(state.result.verdict.first_conflict.hours); };
  $("time").addEventListener("input", () => { setPlaying(false); setHours(Number($("time").value)); });
  $("back").onclick = () => { setPlaying(false); setHours(state.hours - stepHours()); };
  $("fwd").onclick = () => { setPlaying(false); setHours(state.hours + stepHours()); };
  $("play").onclick = () => { if (!state.playing && state.hours >= state.result.verdict.hours) setHours(0); setPlaying(!state.playing); };
  window.addEventListener("keydown", (event) => {
    if (!state.result || ["INPUT", "SELECT"].includes(event.target.tagName) && event.target.id !== "time") return;
    if (event.key === "ArrowLeft") $("back").click();
    else if (event.key === "ArrowRight") $("fwd").click();
    else if (event.key === " ") $("play").click();
    else return;
    event.preventDefault();
  });
  window.addEventListener("resize", resize);

  const pos = (event) => { const box = canvas.getBoundingClientRect(); return [event.clientX - box.left, event.clientY - box.top]; };
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const [sx, sy] = pos(event), [col, row] = toCell(sx, sy), v = state.view;
    v.scale = Math.max(0.05, Math.min(v.scale * Math.exp(-event.deltaY * 0.0015), 12));
    v.ox = sx - col * v.scale; v.oy = sy - row * v.scale; state.dirty = true;
  }, { passive: false });
  canvas.addEventListener("pointerdown", (event) => {
    const [sx, sy] = pos(event);
    const point = state.route.findIndex(([lat, lon]) => { const [x, y] = toScreen(...cellOf(lat, lon)); return Math.hypot(x - sx, y - sy) < 11; });
    state.drag = point >= 0 ? { point } : { sx, sy, ox: state.view.ox, oy: state.view.oy };
    canvas.setPointerCapture(event.pointerId); canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    const [sx, sy] = pos(event), [lat, lon] = latLonOf(...toCell(sx, sy));
    $("cursor").textContent = fmtLatLon(lat, lon);
    if (!state.drag) return;
    if (state.drag.point !== undefined) state.route[state.drag.point] = [lat, lon];
    else { state.view.ox = state.drag.ox + sx - state.drag.sx; state.view.oy = state.drag.oy + sy - state.drag.sy; }
    state.dirty = true;
  });
  const drop = () => {
    if (!state.drag) return;
    const moved = state.drag.point !== undefined;
    state.drag = null; canvas.classList.remove("dragging"); state.dirty = true;
    if (moved) check().catch(showError);
  };
  canvas.addEventListener("pointerup", drop); canvas.addEventListener("pointercancel", drop);
}

wire();
load().then(() => requestAnimationFrame(frame)).catch((error) => { progress(0, `Could not start: ${error.message}`); console.error(error); });
