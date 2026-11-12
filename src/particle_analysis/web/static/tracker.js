// JS port of core/particle_tracking.py ParticleTracker.track_particles (pure; no DOM). Kept step for step with the
// Python so the same detections give the same tracks: tested against the Python output in tracker.test.mjs
// (fixture from scripts/make_engine_fixture.py).
//
// Detections are columns {frame:[], x:[], y:[], intensity:[], sigma:[], snr:[]} in the detector's order.
// Returns tracks [{id, frames:[], x:[], y:[], intensity:[], interp:[bool], start_frame, end_frame}];
// interp marks points the tracker filled in across a gap (linear interpolation, as in Python).

export const TRACKER_DEFAULTS = { max_distance: 5.0, max_gap: 2, min_track_length: 3 };

function byFrame(det) {
  const m = new Map();
  for (let i = 0; i < det.frame.length; i++) {
    const f = det.frame[i];
    if (!m.has(f)) m.set(f, []);
    m.get(f).push({ frame: f, x: det.x[i], y: det.y[i], intensity: det.intensity ? det.intensity[i] : 0, interp: false });
  }
  return m;
}

// All (distance, track index, particle index) with distance <= maxDist, sorted by distance then indices.
// A grid of maxDist-sized cells finds the candidates; distances use the same formula as the Python.
function candidatePairs(tracks, parts, maxDist) {
  const cell = Math.max(maxDist, 1e-9), grid = new Map();
  const key = (cx, cy) => cx + ',' + cy;
  parts.forEach((p, j) => {
    const k = key(Math.floor(p.x / cell), Math.floor(p.y / cell));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(j);
  });
  const pairs = [];
  tracks.forEach((t, i) => {
    const last = t.particles[t.particles.length - 1];
    const cx = Math.floor(last.x / cell), cy = Math.floor(last.y / cell);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const j of grid.get(key(cx + dx, cy + dy)) || []) {
        const ddx = parts[j].x - last.x, ddy = parts[j].y - last.y;
        const d = Math.sqrt(ddx * ddx + ddy * ddy);
        if (d <= maxDist) pairs.push([d, i, j]);
      }
    }
  });
  pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  return pairs;
}

function fillGaps(track, next) {
  const last = track.particles[track.particles.length - 1];
  const gap = next.frame - last.frame;
  for (let f = last.frame + 1; f < next.frame; f++) {
    const fr = (f - last.frame) / gap;
    track.particles.push({ frame: f, x: last.x + fr * (next.x - last.x), y: last.y + fr * (next.y - last.y),
      intensity: last.intensity + fr * (next.intensity - last.intensity), interp: true });
  }
}

export function trackParticles(det, params = {}) {
  const p = { ...TRACKER_DEFAULTS, ...params };
  const frames = byFrame(det);
  const order = [...frames.keys()].sort((a, b) => a - b);
  const tracks = [];
  for (const frame of order) {
    const parts = frames.get(frame);
    if (!tracks.length) {
      for (const q of parts) tracks.push({ id: tracks.length, particles: [q], start_frame: frame, end_frame: frame });
      continue;
    }
    const active = tracks.filter(t => t.end_frame >= frame - p.max_gap - 1);
    const pairs = active.length && parts.length ? candidatePairs(active, parts, p.max_distance) : [];
    const usedP = new Set(), usedT = new Set();
    for (const [, ti, pj] of pairs) {
      if (usedP.has(pj) || usedT.has(ti)) continue;
      const t = active[ti];
      fillGaps(t, parts[pj]);
      t.particles.push(parts[pj]);
      t.end_frame = frame;
      usedP.add(pj); usedT.add(ti);
    }
    parts.forEach((q, j) => {
      if (!usedP.has(j)) tracks.push({ id: tracks.length, particles: [q], start_frame: frame, end_frame: frame });
    });
  }
  return tracks.filter(t => t.particles.length >= p.min_track_length).map(t => ({
    id: t.id, start_frame: t.start_frame, end_frame: t.particles[t.particles.length - 1].frame,
    frames: t.particles.map(q => q.frame), x: t.particles.map(q => q.x), y: t.particles.map(q => q.y),
    intensity: t.particles.map(q => q.intensity), interp: t.particles.map(q => q.interp),
  }));
}
