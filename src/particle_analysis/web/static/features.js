// JS port of core/feature_calculation.py FeatureCalculator (pure; no DOM). Units are those of the track: pixels and
// frames. Tested against the Python output in features.test.mjs (fixture from scripts/make_engine_fixture.py).

export const FEATURE_DEFAULTS = { max_msd_points: 10, min_track_length: 5 };
export const FEATURE_NAMES = ['diffusion_coefficient', 'alpha', 'radius_gyration', 'asymmetry', 'fractal_dimension',
  'straightness', 'mean_velocity', 'mean_acceleration', 'confinement_ratio'];

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

// ordinary least squares y = slope * x + intercept (scipy.stats.linregress / np.polyfit deg 1)
function ols(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

export function msd(pos, times, maxPoints) {
  const n = pos.length, maxLag = Math.min(maxPoints, n - 1);
  const values = [], lags = [];
  for (let lag = 1; lag <= maxLag; lag++) {
    let s = 0, st = 0;
    for (let i = 0; i + lag < n; i++) {
      const dx = pos[i + lag][0] - pos[i][0], dy = pos[i + lag][1] - pos[i][1];
      s += dx * dx + dy * dy; st += times[i + lag] - times[i];
    }
    values.push(s / (n - lag)); lags.push(st / (n - lag));
  }
  return { values, lags };
}

// log-log fit of MSD = 4 D t^alpha over the lags with positive MSD: alpha the slope, D = exp(intercept) / 4.
// Invalid (NaN, NaN) with fewer than two usable lags or a negative slope, as in the Python.
export function fitMsd(values, lags) {
  const x = [], y = [];
  values.forEach((v, i) => { if (v > 0 && lags[i] > 0 && isFinite(v) && isFinite(lags[i])) { x.push(Math.log(lags[i])); y.push(Math.log(v)); } });
  if (x.length < 2) return { D: NaN, alpha: NaN };
  const { slope, intercept } = ols(x, y);
  if (!isFinite(slope) || slope < 0) return { D: NaN, alpha: NaN };
  return { D: Math.exp(intercept) / 4, alpha: slope };
}

export function radiusGyration(pos) {
  const cx = mean(pos.map(p => p[0])), cy = mean(pos.map(p => p[1]));
  return Math.sqrt(mean(pos.map(p => (p[0] - cx) ** 2 + (p[1] - cy) ** 2)));
}

// 1 - lambda_min / lambda_max of the position covariance (ddof 1; the ratio does not depend on it)
export function asymmetry(pos) {
  const n = pos.length, cx = mean(pos.map(p => p[0])), cy = mean(pos.map(p => p[1]));
  let a = 0, b = 0, c = 0;
  for (const [x, y] of pos) { a += (x - cx) ** 2; b += (x - cx) * (y - cy); c += (y - cy) ** 2; }
  a /= n - 1; b /= n - 1; c /= n - 1;
  const h = (a + c) / 2, r = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const lo = h - r, hi = h + r;
  return a + c > 0 ? 1 - lo / hi : 0;
}

// Katz's fractal dimension: log(n) / (log(n) + log(d / L)); n steps, L path length, d largest distance from the
// first point. NaN when undefined (fewer than 2 steps, no movement, or d at most one mean step).
export function fractalDimension(pos) {
  const n = pos.length - 1;
  if (n < 2) return NaN;
  const L = pathLength(pos);
  const d = Math.max(...pos.map(p => Math.sqrt((p[0] - pos[0][0]) ** 2 + (p[1] - pos[0][1]) ** 2)));
  if (!(L > 0) || !(d > 0)) return NaN;
  const den = Math.log(n) + Math.log(d / L);
  return den > 0 ? Math.log(n) / den : NaN;
}

const pathLength = pos => pos.slice(1).reduce((s, p, i) => s + Math.sqrt((p[0] - pos[i][0]) ** 2 + (p[1] - pos[i][1]) ** 2), 0);

export function straightness(pos) {
  if (pos.length < 2) return 0;
  const L = pathLength(pos), e = pos.at(-1);
  return L > 0 ? Math.sqrt((e[0] - pos[0][0]) ** 2 + (e[1] - pos[0][1]) ** 2) / L : 0;
}

export function dynamics(pos, times) {
  if (pos.length < 2) return [0, 0];
  const v = pos.slice(1).map((p, i) => { const dt = times[i + 1] - times[i]; return [(p[0] - pos[i][0]) / dt, (p[1] - pos[i][1]) / dt]; });
  const meanV = mean(v.map(q => Math.sqrt(q[0] * q[0] + q[1] * q[1])));
  if (pos.length <= 2) return [meanV, 0];
  const acc = v.slice(1).map((q, i) => { const dt = times[i + 2] - times[i + 1]; return [(q[0] - v[i][0]) / dt, (q[1] - v[i][1]) / dt]; });
  return [meanV, mean(acc.map(q => Math.sqrt(q[0] * q[0] + q[1] * q[1])))];
}

export function confinementRatio(pos) {
  if (pos.length < 2) return 0;
  const maxD = Math.max(...pos.map(p => Math.sqrt((p[0] - pos[0][0]) ** 2 + (p[1] - pos[0][1]) ** 2)));
  const L = pathLength(pos);
  return L > 0 ? maxD / L : 0;
}

// Features of one track {id, frames, x, y}; null if shorter than min_track_length or with non-finite positions.
export function trackFeatures(t, params = {}) {
  const p = { ...FEATURE_DEFAULTS, ...params };
  if (t.frames.length < p.min_track_length) return null;
  const pos = t.x.map((x, i) => [x, t.y[i]]);
  if (pos.some(q => !isFinite(q[0]) || !isFinite(q[1]))) return null;
  const m = msd(pos, t.frames, p.max_msd_points), fit = fitMsd(m.values, m.lags);
  const [vel, acc] = dynamics(pos, t.frames);
  return { track_id: t.id, msd_values: m.values, diffusion_coefficient: fit.D, alpha: fit.alpha,
    radius_gyration: radiusGyration(pos), asymmetry: asymmetry(pos), fractal_dimension: fractalDimension(pos),
    straightness: straightness(pos), mean_velocity: vel, mean_acceleration: acc, confinement_ratio: confinementRatio(pos) };
}

export const allFeatures = (tracks, params) => tracks.map(t => trackFeatures(t, params)).filter(Boolean);
