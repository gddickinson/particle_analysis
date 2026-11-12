// Features page: FeatureCalculator (JS port) on the current tracks: distributions, a table, the MSD of a chosen track.
import { $, app, download, esc, route } from './core.js';
import { S, ensureData, getFeatures, getInfo, getTracks, truthOf } from './state.js';
import { FEATURE_NAMES } from './features.js';
import { MOTION_COLORS } from './viewer.js';
import { fmt, histogram, lineChart, sortable } from './charts.js';

const LABELS = {
  diffusion_coefficient: ['D (px²/frame)', 'exp(intercept)/(4α) of the log-log MSD fit'],
  alpha: ['α', 'slope of log MSD against log lag: ~1 diffusive, <1 subdiffusive, ~2 ballistic'],
  radius_gyration: ['Rg (px)', 'RMS distance of the points from their centre'],
  asymmetry: ['asymmetry', '1 − λmin/λmax of the position covariance: 0 round, 1 a line'],
  fractal_dimension: ['fractal dim.', 'box-counting slope over 20 scales'],
  straightness: ['straightness', 'end-to-end distance / path length'],
  mean_velocity: ['velocity (px/frame)', 'mean step speed'],
  mean_acceleration: ['acceleration', 'mean |Δv| per frame (px/frame²)'],
  confinement_ratio: ['confinement', 'largest distance from the start / path length'],
};

// Columns of io/writers.py DataWriter.write_features_csv (TrackFeatures.to_dict), msd_values as a list.
function featuresCsv(feats) {
  const cols = ['track_id', 'msd_values', ...FEATURE_NAMES];
  return [cols.join(','), ...feats.map(f => cols.map(c => c === 'msd_values' ? `"${JSON.stringify(f.msd_values)}"` : f[c]).join(','))].join('\n') + '\n';
}

route('features', async () => {
  const info = await getInfo();
  await ensureData();
  const d = S.data, demo = d.id === 'demo';
  const tracks = getTracks(), feats = getFeatures();
  const byId = new Map(tracks.map(t => [t.id, t]));
  if (demo) tracks.forEach(t => { t.truth ??= truthOf(t); });
  const fp = info.defaults.features;
  app().innerHTML = `
    <h1>Track features</h1>
    <p class="lead">FeatureCalculator on the ${tracks.length} tracks from the Track page (tracks with fewer than
      ${fp.min_track_length} points are skipped; the MSD uses up to ${fp.max_msd_points} lags). Units are pixels and frames;
      the Motion page converts to µm and seconds. <span class="badge js">runs in your browser</span></p>
    ${demo ? `<div class="toggles"><label><input type="checkbox" id="bym" checked> split histograms by true motion (simulation)</label></div>` : ''}
    <div class="charts" id="hists"></div>
    <h2>All tracks</h2><div id="table"></div>
    <div class="two"><div class="panel"><h3 style="margin-top:0" id="msdt">MSD</h3><div id="msd"><p class="mute small">Choose a track in the table.</p></div></div>
      <div class="panel"><h3 style="margin-top:0">What the features mean</h3><dl class="kv">${FEATURE_NAMES.map(k => `<dt>${esc(LABELS[k][0])}</dt><dd style="font-family:var(--sans)">${esc(LABELS[k][1])}</dd>`).join('')}</dl></div></div>
    <p><button type="button" class="alt" id="csv">Download features CSV</button><a class="btn" href="#/motion">Next: motion analysis</a></p>`;

  const motions = ['diffusive', 'confined', 'directed', 'immobile'];
  const drawHists = () => {
    const split = demo && $('#bym').checked;
    $('#hists').innerHTML = FEATURE_NAMES.map(k => {
      let body;
      if (split) {
        const series = motions.map(m => ({ m, v: feats.filter(f => byId.get(f.track_id).truth?.motion === m).map(f => f[k]) }));
        body = stripPlot(series, k);
      } else body = histogram(feats.map(f => f[k]), { xlabel: LABELS[k][0] });
      return `<div class="panel"><h3 style="margin-top:0">${esc(LABELS[k][0])}</h3>${body}</div>`;
    }).join('');
  };
  drawHists();
  if (demo) $('#bym').onchange = drawHists;

  sortable($('#table'), [{ key: 'track_id', label: 'Track' }, { key: 'n', label: 'Points' },
    ...(demo ? [{ key: 'truth', label: 'True motion', fmt: v => v ? `<span style="color:${MOTION_COLORS[v]}">${v}</span>` : '–' }] : []),
    ...FEATURE_NAMES.map(k => ({ key: k, label: LABELS[k][0], fmt: v => fmt(v) }))],
  feats.map(f => ({ ...f, n: byId.get(f.track_id).frames.length, truth: byId.get(f.track_id).truth?.motion })), {
    onRow: r => {
      $('#msdt').textContent = `MSD of track ${r.track_id}`;
      const lags = r.msd_values.map((_, i) => i + 1);
      const fit = lags.map(t => 4 * r.diffusion_coefficient * r.alpha * t ** r.alpha);
      $('#msd').innerHTML = lineChart([{ x: lags, y: r.msd_values, color: '#7cff6b', points: true, label: 'MSD' },
        { x: lags, y: fit, color: '#ff5ad1', dash: true, label: `fit: α = ${fmt(r.alpha, 2)}` }], { xlabel: 'lag (frames)', ylabel: 'MSD (px²)' });
    },
  });
  $('#csv').onclick = () => download(`${(d.title || 'tracks').replace(/\.[^.]+$/, '').replace(/\W+/g, '_')}_features.csv`, featuresCsv(feats), 'text/csv');
});

// One row of dots per group (values on a shared x axis): readable for a handful of tracks per group.
function stripPlot(series, key) {
  const all = series.flatMap(s => s.v).filter(v => v !== null && isFinite(v));
  if (!all.length) return '<p class="mute small">No finite values.</p>';
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { lo -= 0.5; hi += 0.5; }
  const W = 420, L = 78, R = 12, rowH = 26, H = series.length * rowH + 30, sx = v => L + (v - lo) / (hi - lo) * (W - L - R);
  const rows = series.map((s, i) => `<text x="${L - 8}" y="${i * rowH + 20}" text-anchor="end">${s.m}</text>` +
    s.v.filter(v => v !== null && isFinite(v)).map(v => `<circle cx="${sx(v)}" cy="${i * rowH + 16}" r="4" fill="${MOTION_COLORS[s.m]}" opacity=".85"><title>${fmt(v)}</title></circle>`).join('')).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(key)} by true motion type">${rows}
    <line class="axis" x1="${L}" y1="${H - 24}" x2="${W - R}" y2="${H - 24}"/>
    <text x="${L}" y="${H - 8}">${fmt(lo, 2)}</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${fmt(hi, 2)}</text></svg>`;
}
