// App state: the current dataset (demo recording or the user's data), the tracker settings, and memoised tracks and
// features computed in the browser by the JS ports.
import { $, api, isStatic } from './core.js';
import { trackParticles } from './tracker.js';
import { allFeatures } from './features.js';

export const S = { info: null, data: null, tracker: null, version: 0, memo: {} };

export async function getInfo() {
  if (!S.info) {
    S.info = await api('/info');
    S.tracker = { ...S.info.defaults.tracker };
  }
  return S.info;
}

function decodeMovie(m) {
  if (!m || m.frames) return m;
  const bin = atob(m.data), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const size = m.width * m.height;
  m.frames = Array.from({ length: m.n }, (_, k) => u8.subarray(k * size, (k + 1) * size));
  delete m.data;
  return m;
}

function setData(d) {
  S.data = d; S.version++; S.memo = {};
  const el = $('#dataset');
  if (el) el.textContent = d ? (d.simulated ? 'Demo (simulated)' : d.title) : '';
}

export async function loadDemo(threshold) {
  const info = await getInfo();
  const thr = threshold || info.demo.default_threshold;
  if (S.data && S.data.id === 'demo' && S.data.threshold === thr) return S.data;
  const mv = await api('/demo/movie');
  decodeMovie(mv.movie);
  const det = await api('/demo/detections/' + thr);
  setData({ ...info.demo, movie: mv.movie, truth: mv.truth, threshold: thr, params: det.params,
    detections: det.detections, n: det.n, frame_interval_source: 'simulation parameters' });
  return S.data;
}

export async function ensureData() {
  if (!S.data) await loadDemo();
  return S.data;
}

// Result of POST /api/detect (local app) as the current dataset.
export function setUpload(r, frameInterval) {
  decodeMovie(r.movie);
  const info = S.info;
  setData({ ...r, frame_interval_s: frameInterval ?? info.defaults.diffusion.frame_interval,
    frame_interval_source: frameInterval ? 'entered on the Your data page' : 'DiffusionAnalyzer default (not from the file)',
    pixel_size_um: r.pixel_size_um ?? info.defaults.diffusion.pixel_size,
    pixel_size_source: r.pixel_size_um ? r.pixel_size_source : 'DiffusionAnalyzer default (not in the file)' });
}

// A detections table (CSV) as the current dataset: no movie, tracking and features only.
export function setDetections(name, det) {
  const info = S.info;
  setData({ id: 'csv', title: name, simulated: false, movie: null, truth: null, detections: det, n: det.frame.length,
    params: null, pixel_size_um: info.defaults.diffusion.pixel_size, pixel_size_source: 'DiffusionAnalyzer default (not in the file)',
    frame_interval_s: info.defaults.diffusion.frame_interval, frame_interval_source: 'DiffusionAnalyzer default (not in the file)' });
}

export const trackerKey = () => JSON.stringify(S.tracker);
export const trackerIsDefault = () => trackerKey() === JSON.stringify(S.info.defaults.tracker);

export function getTracks() {
  const k = 'tracks' + trackerKey();
  if (!S.memo[k]) S.memo = { [k]: trackParticles(S.data.detections, S.tracker) };
  return S.memo[k];
}

export function getFeatures() {
  getTracks();
  const k = 'features' + trackerKey();
  if (!S.memo[k]) S.memo[k] = allFeatures(getTracks(), S.info.defaults.features);
  return S.memo[k];
}

// Demo only: the true particle each track follows (nearest true position, majority over its detected points).
export function truthOf(track) {
  const truth = S.data && S.data.truth;
  if (!truth) return null;
  const votes = new Map();
  track.frames.forEach((f, i) => {
    if (track.interp && track.interp[i]) return;
    let best = null, bd = 2.0;
    for (const q of truth) {
      if (f >= q.x.length) continue;
      const d = Math.hypot(q.x[f] - track.x[i], q.y[f] - track.y[i]);
      if (d < bd) { bd = d; best = q; }
    }
    if (best) votes.set(best, (votes.get(best) || 0) + 1);
  });
  let top = null, n = 0;
  for (const [q, c] of votes) if (c > n) { top = q; n = c; }
  return top ? { id: top.id, motion: top.motion, share: n / track.frames.length } : null;
}

export const modeNote = () => isStatic()
  ? 'static site: demo detections precomputed by the Python code; tracking and features run in your browser'
  : 'local app: detection and motion analysis run in Python on this computer; tracking and features in the browser';
