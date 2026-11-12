// Motion page: DiffusionAnalyzer (Python) on the current tracks: MSD in µm² against lag in s, the four model fits and
// the motion type. Precomputed for the demo with the default tracker; otherwise run by the local server.
import { $, api, app, esc, isStatic, needsServer, route } from './core.js';
import { S, ensureData, getInfo, getTracks, trackerIsDefault, truthOf } from './state.js';
import { MOTION_COLORS } from './viewer.js';
import { fmt, lineChart, sortable } from './charts.js';

const TYPES = ['normal_diffusion', 'confined', 'directed', 'anomalous'];
const NC = 'unclassified';
const typeLabel = v => (v || NC).replace('_', ' ');
const TRUE_TYPES = ['diffusive', 'confined', 'directed', 'immobile'];
const MODEL_COLORS = { normal: '#5ab8ff', anomalous: '#ff8a3d', confined: '#ffd23f', directed: '#ff5ad1' };

async function analyse(px, dt) {
  const d = S.data, info = S.info;
  const demoDefault = d.id === 'demo' && trackerIsDefault() && px === d.pixel_size_um && dt === d.frame_interval_s;
  if (demoDefault) {
    const r = await api('/demo/motion/' + d.threshold);
    return { ...r, source: `precomputed by the Python code${isStatic() ? ' when the site was built' : ''}` };
  }
  const tracks = getTracks().map(t => ({ id: t.id, frames: t.frames, x: t.x, y: t.y }));
  const r = await api('/motion', { body: { tracks, pixel_size: px, frame_interval: dt } });
  return { ...r, source: 'computed now by the Python server', units_source: 'from the fields above' };
}

route('motion', async () => {
  await getInfo();
  await ensureData();
  const d = S.data, demo = d.id === 'demo';
  app().innerHTML = `
    <h1>Motion analysis</h1>
    <p class="lead">DiffusionAnalyzer converts each track to µm and seconds, computes the MSD for lags up to a quarter of
      the track and fits four models by generalised least squares (weighted by the covariance of the MSD of a Brownian
      track, since the lags of one track are correlated). The model with the lowest BIC wins, which penalises the extra
      parameter of the anomalous, confined and directed models; if it is the anomalous one, α &lt; 0.8 is called confined
      and α &gt; 1.2 directed. Tracks with fewer than 4 MSD points are unclassified. <span class="badge py">Python</span></p>
    <p class="eq">normal 4Dt · anomalous 4Dt<sup>α</sup> · confined A(1 − e<sup>−4Dt/A</sup>) · directed 4Dt + (vt)²</p>
    <form class="panel params row" id="units">
      <div><label for="px">Pixel size (µm)</label><input id="px" type="number" step="any" min="0.0001" value="${d.pixel_size_um}">
        <p class="hint">${esc(d.pixel_size_source)}</p></div>
      <div><label for="dt">Frame interval (s)</label><input id="dt" type="number" step="any" min="0.00001" value="${d.frame_interval_s}">
        <p class="hint">${esc(d.frame_interval_source)}</p></div>
      <div><button type="submit">Analyse</button></div>
    </form>
    <div id="out"><p class="mute">Loading...</p></div>`;

  const run = async () => {
    const px = +$('#px').value, dt = +$('#dt').value, out = $('#out');
    out.innerHTML = '<p class="mute">Fitting...</p>';
    let r;
    try { r = await analyse(px, dt); } catch (e) {
      out.innerHTML = isStatic() ? `${needsServer('The motion analysis of these tracks')}<p class="mute small">On this static site it is
        precomputed for the demo recording with the default tracker settings and the simulation's units. ${demo ? '<a href="#/tracks">Reset the tracker to its defaults</a> to see it.' : ''}</p>`
        : `<p class="err">${esc(e.message)}</p>`;
      return;
    }
    render(out, r, d, demo);
  };
  $('#units').onsubmit = e => { e.preventDefault(); run(); };
  run();
});

function render(out, r, d, demo) {
  const tracks = new Map(getTracks().map(t => [t.id, t]));
  const rows = r.motion.map(m => {
    const t = tracks.get(m.track_id) || {};
    if (demo && t.frames) t.truth ??= truthOf(t);
    const best = Object.entries(m.fits).filter(([, f]) => f.bic !== null).sort((a, b) => a[1].bic - b[1].bic)[0];
    return { track_id: m.track_id, n: m.msd.length, type: m.motion_type || NC, best: best ? best[0] : '–', bic: best ? best[1].bic : null,
      D: m.fits.normal?.D, alpha: m.fits.anomalous?.alpha, truth: t.truth?.motion, m };
  });
  const counts = Object.fromEntries([...TYPES, NC].map(k => [k, rows.filter(x => x.type === k).length]));
  out.innerHTML = `
    <p class="mute small">${r.motion.length} tracks; pixel size ${r.pixel_size_um} µm, frame interval ${r.frame_interval_s} s
      (${esc(r.units_source || '')}); ${esc(r.source)}.</p>
    <div class="tiles">${TYPES.map(k => `<div class="tile" style="border-left-color:${MOTION_COLORS[k]}"><b>${counts[k]}</b><span>${k.replace('_', ' ')}</span></div>`).join('')}
      ${counts[NC] ? `<div class="tile" style="border-left-color:var(--mute)"><b>${counts[NC]}</b><span>${NC} (too short or fits failed)</span></div>` : ''}</div>
    ${demo ? `<div class="panel"><h3 style="margin-top:0">Classified type against the simulated motion</h3>${confusion(rows)}
      <p class="hint">Rows: how the particle was simulated; columns: what the classifier called its track. Immobile
      particles have no model of their own: their MSD is a localisation-noise plateau, which the confined model fits.
      Short tracks are classified less reliably (in tests on simulated 100-frame tracks, about 87% of diffusive and
      confined and all directed tracks came out right).</p></div>` : ''}
    <h2>Tracks</h2><div id="mtable"></div>
    <div class="panel"><h3 style="margin-top:0" id="fitt">Model fits</h3><div id="fit"><p class="mute small">Choose a track in the table.</p></div></div>`;
  sortable($('#mtable'), [{ key: 'track_id', label: 'Track' }, { key: 'n', label: 'MSD lags' },
    { key: 'type', label: 'Motion type', fmt: v => `<span style="color:${MOTION_COLORS[v] || 'var(--mute)'}">${typeLabel(v === NC ? null : v)}</span>` },
    { key: 'best', label: 'Lowest BIC' }, { key: 'bic', label: 'BIC', fmt: v => fmt(v, 1) },
    { key: 'D', label: 'D normal (µm²/s)', fmt: v => fmt(v, 4) }, { key: 'alpha', label: 'α anomalous', fmt: v => fmt(v, 2) },
    ...(demo ? [{ key: 'truth', label: 'Simulated as', fmt: v => v ? `<span style="color:${MOTION_COLORS[v]}">${v}</span>` : '–' }] : [])],
  rows, { onRow: x => showFit(x.m) });
}

function confusion(rows) {
  const cell = (t, k) => rows.filter(x => x.truth === t && x.type === k).length;
  return `<div class="scroll"><table class="tbl cm"><thead><tr><th>simulated \\ classified</th>${[...TYPES, NC].map(k => `<th>${k.replace('_', ' ')}</th>`).join('')}</tr></thead>
    <tbody>${TRUE_TYPES.map(t => `<tr><td style="color:${MOTION_COLORS[t]}">${t}</td>${[...TYPES, NC].map(k => `<td class="n">${cell(t, k) || '·'}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function showFit(m) {
  $('#fitt').textContent = `Model fits, track ${m.track_id} (${typeLabel(m.motion_type)})`;
  const series = [{ x: m.time_lags, y: m.msd, color: '#7cff6b', points: true, label: 'MSD' },
    ...Object.entries(m.fits).map(([k, f]) => ({ x: m.time_lags, y: f.fit_values, color: MODEL_COLORS[k], dash: true, label: `${k} (BIC ${fmt(f.bic, 1)})` }))];
  const params = Object.entries(m.fits).map(([k, f]) => `<dt>${k}</dt><dd>${Object.entries(f).filter(([p]) => !['fit_values', 'r_squared', 'chi2', 'bic', 'n_params'].includes(p)).map(([p, v]) => `${p} = ${fmt(v, 4)}`).join(', ')}</dd>`).join('');
  $('#fit').innerHTML = `<div class="two"><div>${lineChart(series, { xlabel: 'lag (s)', ylabel: 'MSD (µm²)' })}</div><dl class="kv">${params}</dl></div>
    <p class="hint">Units: D in µm²/s, v in µm/s, confinement_size = √A in µm.</p>`;
}
