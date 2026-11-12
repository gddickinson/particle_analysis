// Track page: the JS tracker (port of ParticleTracker) on the current detections, with live settings, tracks drawn
// over the recording, a table of tracks and (demo) how well tracks follow the simulated particles.
import { $, app, download, esc, route, slider } from './core.js';
import { S, ensureData, getInfo, getTracks, truthOf } from './state.js';
import { COLORS, MOTION_COLORS, createViewer, ring } from './viewer.js';
import { fmt, histogram, sortable } from './charts.js';

// Same first columns as io/writers.py DataWriter.write_tracks_csv (readable by DataReader.read_tracks_csv),
// plus 'interpolated' for points the tracker filled in across a gap.
export function tracksCsv(tracks) {
  const rows = ['track_number,frame,x,y,intensity,start_frame,end_frame,interpolated'];
  for (const t of tracks) t.frames.forEach((f, i) => rows.push([t.id, f, t.x[i], t.y[i], t.intensity[i], t.start_frame, t.end_frame, t.interp[i] ? 1 : 0].join(',')));
  return rows.join('\n') + '\n';
}

route('tracks', async () => {
  const info = await getInfo();
  await ensureData();
  const d = S.data, demo = d.id === 'demo';
  app().innerHTML = `
    <h1>Link detections into tracks</h1>
    <p class="lead">Each frame, every (track, detection) pair closer than the maximum distance is a candidate; the closest
      pairs are linked first. A track may skip up to <i>max gap</i> frames (the missing points are interpolated) and is kept
      if it has at least the minimum number of points. <span class="badge js">runs in your browser</span></p>
    <div class="side">
      <div>
        <div class="panel params"><h3 style="margin-top:0">Tracker settings</h3><div id="ctl"></div>
          <button type="button" class="alt" id="reset">Defaults</button>
          <p class="hint">Defaults from ParticleTracker: ${Object.entries(info.defaults.tracker).map(([k, v]) => `${k} ${v}`).join(', ')}.</p></div>
        <div class="tiles" id="tiles" style="grid-template-columns:1fr 1fr"></div>
        <div class="panel" id="truth" ${demo ? '' : 'hidden'}></div>
      </div>
      <div>
        <div class="toggles"><label><input type="checkbox" id="t-trail" checked> trails up to this frame</label>
          <label><input type="checkbox" id="t-all" checked> whole tracks</label>
          ${demo ? '<label><input type="checkbox" id="t-motion"> colour by true motion</label>' : ''}</div>
        <div id="view"></div>
        <p class="mute small">Dashed segments: gap-filled (interpolated) points.${demo ? ' Colours by true motion: ' + Object.entries({ diffusive: 1, confined: 1, directed: 1, immobile: 1 }).map(([k]) => `<span style="color:${MOTION_COLORS[k]}">${k}</span>`).join(', ') + '.' : ''}</p>
      </div>
    </div>
    <div class="charts"><div class="panel"><h3 style="margin-top:0">Track lengths</h3><div id="lenhist"></div></div></div>
    <h2>Tracks</h2><div id="table"></div>
    <p><button type="button" class="alt" id="csv">Download tracks CSV</button><a class="btn" href="#/features">Next: track features</a></p>`;

  const ctl = $('#ctl'), sliders = {};
  const spec = [['max_distance', 'Max distance (px)', 0.5, 20, 0.5], ['max_gap', 'Max gap (frames)', 0, 10, 1], ['min_track_length', 'Min track length', 2, 30, 1]];
  for (const [k, label, min, max, step] of spec) sliders[k] = slider(ctl, { id: k, label, min, max, step, value: S.tracker[k] }, v => { S.tracker[k] = v; update(); });
  $('#reset').onclick = () => { S.tracker = { ...info.defaults.tracker }; for (const [k] of spec) { sliders[k].value = S.tracker[k]; sliders[k].dispatchEvent(new Event('input')); } };

  let tracks = [], selected = null;
  const showTrail = $('#t-trail'), showAll = $('#t-all'), byMotion = $('#t-motion');
  const color = (t, i) => (byMotion && byMotion.checked && t.truth) ? MOTION_COLORS[t.truth.motion] : COLORS[i % COLORS.length];
  const viewer = createViewer($('#view'), d, (g, f, at) => {
    tracks.forEach((t, i) => {
      if (f < t.start_frame || (f > t.end_frame && !showAll.checked)) return;
      const c = color(t, i), end = showAll.checked ? t.frames.length : t.frames.findIndex(x => x > f);
      const upto = end < 0 ? t.frames.length : end;
      if (showTrail.checked || showAll.checked) {
        g.lineWidth = t === selected ? 3 : 1.5;
        for (let k = 1; k < upto; k++) {
          g.strokeStyle = c; g.setLineDash(t.interp[k] || t.interp[k - 1] ? [3, 3] : []);
          g.beginPath(); g.moveTo(...at(t.x[k - 1], t.y[k - 1])); g.lineTo(...at(t.x[k], t.y[k])); g.stroke();
        }
        g.setLineDash([]);
      }
      const k = t.frames.indexOf(f);
      if (k >= 0) ring(g, at(t.x[k], t.y[k]), 6, c, t === selected ? 3 : 1.8);
    });
  });
  [showTrail, showAll, byMotion].forEach(el => el && (el.onchange = () => viewer.render()));

  function update() {
    tracks = getTracks();
    if (demo) tracks.forEach(t => { t.truth ??= truthOf(t); });
    const nInterp = tracks.reduce((s, t) => s + t.interp.filter(Boolean).length, 0);
    const linked = tracks.reduce((s, t) => s + t.interp.filter(x => !x).length, 0);
    $('#tiles').innerHTML = `<div class="tile"><b>${tracks.length}</b><span>tracks</span></div>
      <div class="tile"><b>${tracks.length ? fmt(tracks.reduce((s, t) => s + t.frames.length, 0) / tracks.length, 1) : '–'}</b><span>mean length (points)</span></div>
      <div class="tile"><b>${linked}/${d.n}</b><span>detections in kept tracks</span></div>
      <div class="tile"><b>${nInterp}</b><span>gap-filled points</span></div>`;
    $('#lenhist').innerHTML = histogram(tracks.map(t => t.frames.length), { xlabel: 'points per track' });
    if (demo) {
      const per = d.truth.map(q => tracks.filter(t => t.truth && t.truth.id === q.id));
      const whole = per.filter(ts => ts.length === 1 && ts[0].frames.length >= d.n_frames * 0.9).length;
      $('#truth').innerHTML = `<h3 style="margin-top:0">Against the ground truth</h3><dl class="kv">
        <dt>followed end to end</dt><dd>${whole} of ${d.truth.length} particles (one track, ≥90% of frames)</dd>
        <dt>split</dt><dd>${per.filter(ts => ts.length > 1).length} particles in 2+ tracks</dd>
        <dt>missed</dt><dd>${per.filter(ts => !ts.length).length} particles with no track</dd>
        <dt>unmatched tracks</dt><dd>${tracks.filter(t => !t.truth).length}</dd></dl>
        <p class="hint">Each track is assigned to the simulated particle most of its detected points lie on (within 2 px).</p>`;
    }
    sortable($('#table'), [
      { key: 'id', label: 'Track' }, { key: 'start_frame', label: 'Start' }, { key: 'end_frame', label: 'End' },
      { key: 'n', label: 'Points' }, { key: 'gaps', label: 'Gap-filled' },
      { key: 'disp', label: 'Net displacement (px)', fmt: v => fmt(v, 2) },
      ...(demo ? [{ key: 'truthLabel', label: 'True particle' }] : []),
    ], tracks.map(t => ({ id: t.id, start_frame: t.start_frame, end_frame: t.end_frame, n: t.frames.length,
      gaps: t.interp.filter(Boolean).length, disp: Math.hypot(t.x.at(-1) - t.x[0], t.y.at(-1) - t.y[0]),
      truthLabel: t.truth ? `#${t.truth.id} ${t.truth.motion}` : '–', t })), {
      onRow: r => { selected = r.t; viewer.set(r.t.start_frame); },
    });
    viewer.render();
  }
  $('#csv').onclick = () => download(`${(d.title || 'tracks').replace(/\.[^.]+$/, '').replace(/\W+/g, '_')}_tracks.csv`, tracksCsv(tracks), 'text/csv');
  update();
});
