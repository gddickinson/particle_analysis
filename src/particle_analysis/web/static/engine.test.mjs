// Node test: the JS tracker and feature ports reproduce the Python output (tests/fixtures/engine_cases.json,
// written by scripts/make_engine_fixture.py). Run: node src/particle_analysis/web/static/engine.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { trackParticles } from './tracker.js';
import { allFeatures, FEATURE_NAMES } from './features.js';

const fixture = JSON.parse(readFileSync(new URL('../../../../tests/fixtures/engine_cases.json', import.meta.url)));
const close = (a, b, what) => {
  if (a === null || b === null || Number.isNaN(a) || Number.isNaN(b)) {
    assert.ok((a === null || Number.isNaN(a)) && (b === null || Number.isNaN(b)), `${what}: ${a} vs ${b}`);
    return;
  }
  assert.ok(Math.abs(a - b) <= 1e-9 + 1e-7 * Math.abs(b), `${what}: js ${a} vs python ${b}`);
};

let nTracks = 0, nFeat = 0;
for (const c of fixture.cases) {
  const js = trackParticles(fixture.inputs[c.input], c.tracker);
  const label = `${c.name} ${JSON.stringify(c.tracker)}`;
  assert.equal(js.length, c.tracks.length, `${label}: number of tracks`);
  js.forEach((t, i) => {
    const py = c.tracks[i];
    assert.equal(t.id, py.id, `${label}: track id`);
    assert.deepEqual(t.frames, py.frames, `${label}: frames of track ${py.id}`);
    t.x.forEach((x, k) => { close(x, py.x[k], `${label} track ${py.id} x`); close(t.y[k], py.y[k], `${label} track ${py.id} y`); });
  });
  nTracks += js.length;
  const feats = allFeatures(js);
  assert.equal(feats.length, c.features.length, `${label}: number of feature rows`);
  feats.forEach((f, i) => {
    const py = c.features[i];
    assert.equal(f.track_id, py.track_id);
    for (const k of FEATURE_NAMES) close(f[k], py[k], `${label} track ${py.track_id} ${k}`);
    f.msd_values.forEach((v, k) => close(v, py.msd_values[k], `${label} track ${py.track_id} msd[${k}]`));
  });
  nFeat += feats.length;
}
console.log(`ok: ${fixture.cases.length} cases, ${nTracks} tracks and ${nFeat} feature rows match the Python output`);
