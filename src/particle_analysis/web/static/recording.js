// Detect page: the recording with the detector's output (and, for the simulated demo, the true positions), the
// detection threshold, per-frame counts and the accuracy against the ground truth.
import { $, app, esc, isStatic, route } from './core.js';
import { S, ensureData, getInfo, loadDemo } from './state.js';
import { createViewer, cross, ring } from './viewer.js';
import { fmt, histogram, lineChart } from './charts.js';

const MATCH_PX = 1.5;

function byFrame(det) {
  const m = [];
  det.frame.forEach((f, i) => { (m[f] ||= []).push(i); });
  return m;
}

// Demo: detections matched to visible true particles (nearest, within MATCH_PX, one-to-one greedy by distance)
function accuracy(data, frames) {
  let tp = 0, fp = 0, fn = 0;
  const err = [];
  data.truth[0].x.forEach((_, f) => {
    const idx = frames[f] || [], pairs = [];
    data.truth.forEach((q, qi) => {
      if (!q.visible[f]) return;
      idx.forEach(i => { const d = Math.hypot(data.detections.x[i] - q.x[f], data.detections.y[i] - q.y[f]); if (d < MATCH_PX) pairs.push([d, qi, i]); });
    });
    pairs.sort((a, b) => a[0] - b[0]);
    const uq = new Set(), ud = new Set();
    for (const [d, qi, i] of pairs) if (!uq.has(qi) && !ud.has(i)) { uq.add(qi); ud.add(i); err.push(d); }
    const vis = data.truth.filter(q => q.visible[f]).length;
    tp += uq.size; fn += vis - uq.size; fp += idx.length - ud.size;
  });
  return { tp, fp, fn, recall: tp / (tp + fn), precision: tp / (tp + fp), rms: Math.sqrt(err.reduce((s, e) => s + e * e, 0) / err.length) };
}

route('recording', async (args, query) => {
  const info = await getInfo();
  if (query.thr) await loadDemo(query.thr); else await ensureData();
  const d = S.data, det = d.detections, frames = byFrame(det);
  const demo = d.id === 'demo';
  app().innerHTML = `
    <h1>Detect particles</h1>
    <p class="lead">${demo ? `<span class="badge sim">simulated</span> ${esc(d.description)}` : `${esc(d.title)}: ${d.n.toLocaleString()} detections.`}</p>
    <div class="side">
      <div>
        ${demo ? `<div class="panel params"><h3 style="margin-top:0">Detection threshold</h3>
          <p class="hint">threshold_rel: peaks must reach this fraction of the frame's brightest background-subtracted pixel.
          Precomputed by the Python detector${isStatic() ? ' when the site was built' : ''}; other parameters at their defaults.</p>
          <div class="seg" role="group" aria-label="Detection threshold">${info.demo.thresholds.map(t => `<button type="button" data-thr="${t}" class="${t === d.threshold ? 'on' : ''}">${t}</button>`).join('')}</div></div>` : ''}
        <div class="panel"><h3 style="margin-top:0">Detector parameters</h3>
          ${d.params ? `<dl class="kv">${Object.entries(d.params).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`
            : '<p class="mute small">Detections loaded from a table; the detector did not run here.</p>'}</div>
        <div class="panel" id="acc"></div>
      </div>
      <div>
        <div class="toggles">
          <label><input type="checkbox" id="t-det" checked> <span class="key" style="--c:#ff5ad1"></span> detections</label>
          ${demo ? '<label><input type="checkbox" id="t-truth"> <span class="key x" style="--c:#5ab8ff"></span> true positions</label>' : ''}
        </div>
        <div id="view"></div>
        <p class="mute small" id="fcount"></p>
      </div>
    </div>
    <div class="charts">
      <div class="panel"><h3 style="margin-top:0">Detections per frame</h3><div id="perframe"></div></div>
      <div class="panel"><h3 style="margin-top:0">Signal-to-noise ratio</h3>${histogram(det.snr || [], { xlabel: 'SNR (fit amplitude / frame noise)' })}</div>
      <div class="panel"><h3 style="margin-top:0">Fitted spot width</h3>${histogram(det.sigma || [], { xlabel: 'sigma (px)' })}</div>
    </div>
    <p><a class="btn" href="#/tracks">Next: link into tracks</a></p>`;

  app().querySelectorAll('[data-thr]').forEach(b => b.onclick = () => { location.hash = '#/recording?thr=' + b.dataset.thr; });
  const showDet = $('#t-det'), showTruth = $('#t-truth');
  const viewer = createViewer($('#view'), d, (g, f, at) => {
    const idx = frames[f] || [];
    if (showTruth && showTruth.checked) d.truth.forEach(q => { if (q.visible[f]) cross(g, at(q.x[f], q.y[f]), 4, '#5ab8ff'); });
    if (showDet.checked) idx.forEach(i => ring(g, at(det.x[i], det.y[i]), Math.max(5, 2.2 * (det.sigma ? det.sigma[i] : 1.5) * at.scale), '#ff5ad1'));
    $('#fcount').textContent = `Frame ${f + 1}: ${idx.length} detections` +
      (demo ? `, ${d.truth.filter(q => q.visible[f]).length} particles visible in the simulation` : '');
  });
  [showDet, showTruth].forEach(el => el && (el.onchange = () => viewer.render()));

  const nF = d.movie ? d.movie.n_total || d.movie.n : Math.max(...det.frame) + 1;
  const counts = Array.from({ length: nF }, (_, f) => (frames[f] || []).length);
  $('#perframe').innerHTML = lineChart([{ x: counts.map((_, i) => i + 1), y: counts, color: '#ff5ad1', label: 'detected' },
    ...(demo ? [{ x: counts.map((_, i) => i + 1), y: counts.map((_, f) => d.truth.filter(q => q.visible[f]).length), color: '#5ab8ff', dash: true, label: 'visible (truth)' }] : [])],
    { xlabel: 'frame', ylabel: 'count' });

  const acc = $('#acc');
  if (demo) {
    const a = accuracy(d, frames);
    acc.innerHTML = `<h3 style="margin-top:0">Against the ground truth</h3>
      <dl class="kv"><dt>found</dt><dd>${a.tp} of ${a.tp + a.fn} visible (${fmt(100 * a.recall, 1)}%)</dd>
      <dt>false detections</dt><dd>${a.fp} (precision ${fmt(100 * a.precision, 1)}%)</dd>
      <dt>position error</dt><dd>${fmt(a.rms, 3)} px RMS (${fmt(a.rms * d.pixel_size_um * 1000, 0)} nm)</dd></dl>
      <p class="hint">A detection counts as finding a particle when it lies within ${MATCH_PX} px of its true position (one-to-one, closest first). Computed in your browser from the detections and the simulation's ground truth.</p>`;
  } else {
    acc.innerHTML = `<h3 style="margin-top:0">Recording</h3><dl class="kv">
      ${d.shape ? `<dt>frames used</dt><dd>${d.n_frames_used} of ${d.n_frames_total}</dd><dt>size</dt><dd>${d.shape[2]}×${d.shape[1]} px (${esc(d.dtype)})</dd>` : ''}
      <dt>pixel size</dt><dd>${d.pixel_size_um} µm (${esc(d.pixel_size_source)})</dd>
      ${d.movie && d.movie.bin > 1 ? `<dt>display</dt><dd>binned ${d.movie.bin}×${d.movie.bin} for viewing only</dd>` : ''}</dl>`;
  }
});
