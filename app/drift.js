"use strict";
// AI-generated: the code of this interface was written with Claude (Anthropic), an AI assistant, directed and checked by the team.
// Ice drift: how far ice has moved between two times. The SAME formula as scripts/route_safety.py (displacement):
// the drift of each 48 h window (km/day along the map x and y axes, on a 10 km grid, bilinear between nodes) times
// the part of the window between the two times; negative when going back in time. Outside the windows: no motion.
// drift = { x0, y0, node_m, nx, ny, edges_s: [UNIX seconds], data: Float32Array (window, u/v, row, col) }

const Drift = (() => {
  function velocity(drift, x, y, window) {
    const { nx, ny, node_m: step, data } = drift;
    const fx = Math.min(Math.max((x - drift.x0) / step, 0), nx - 1.000001);
    const fy = Math.min(Math.max((drift.y0 - y) / step, 0), ny - 1.000001);
    const i = Math.floor(fx), j = Math.floor(fy), a = fx - i, b = fy - j;
    const at = (comp, jj, ii) => data[((window * 2 + comp) * ny + jj) * nx + ii];
    const mix = (comp) => at(comp, j, i) * (1 - a) * (1 - b) + at(comp, j, i + 1) * a * (1 - b) + at(comp, j + 1, i) * (1 - a) * b + at(comp, j + 1, i + 1) * a * b;
    return [mix(0), mix(1)];
  }

  function displacement(drift, x, y, tFrom, tTo) {        // metres along map x, y
    const lo = Math.min(tFrom, tTo), hi = Math.max(tFrom, tTo), sign = Math.sign(tTo - tFrom), e = drift.edges_s;
    let dx = 0, dy = 0;
    for (let w = 0; w < e.length - 1; w++) {
      const days = Math.max(Math.min(hi, e[w + 1]) - Math.max(lo, e[w]), 0) / 86400;
      if (days <= 0) continue;
      const [u, v] = velocity(drift, x, y, w);
      dx += u * days; dy += v * days;
    }
    return [sign * dx * 1000, sign * dy * 1000];
  }

  function windowAt(drift, t) {                          // index of the 48 h window that holds time t, or -1
    const e = drift.edges_s;
    for (let w = 0; w < e.length - 1; w++) if (t >= e[w] && t < e[w + 1]) return w;
    return -1;
  }

  return { velocity, displacement, windowAt };
})();

if (typeof module !== "undefined") module.exports = Drift;
