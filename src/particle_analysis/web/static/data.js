// Your data: a TIFF stack analysed by the Python detector (local app), or a detections table tracked in the browser.
import { $, api, app, esc, isStatic, needsServer, route } from './core.js';
import { S, getInfo, loadDemo, setDetections, setUpload } from './state.js';

// CSV with columns frame, x, y (+ optional intensity, sigma, snr), as io/writers.py write_particles_csv writes.
export function parseDetectionsCsv(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) throw new Error('The file has no data rows.');
  const head = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, '').toLowerCase());
  const col = n => head.indexOf(n);
  for (const n of ['frame', 'x', 'y']) if (col(n) < 0) throw new Error(`Missing column "${n}" (need frame, x, y; found: ${head.join(', ')}).`);
  const out = { frame: [], x: [], y: [], intensity: [], sigma: [], snr: [] };
  lines.slice(1).forEach((l, i) => {
    const c = l.split(',');
    const f = Number(c[col('frame')]), x = Number(c[col('x')]), y = Number(c[col('y')]);
    if (!Number.isInteger(f) || f < 0 || !isFinite(x) || !isFinite(y)) throw new Error(`Row ${i + 2}: frame must be a whole number ≥ 0 and x, y numbers.`);
    out.frame.push(f); out.x.push(x); out.y.push(y);
    for (const k of ['intensity', 'sigma', 'snr']) out[k].push(col(k) >= 0 ? Number(c[col(k)]) : (k === 'sigma' ? 1.5 : 0));
  });
  return out;
}

route('data', async () => {
  const info = await getInfo();
  const det = info.defaults.detector, lim = info.limits;
  const num = (k, step) => `<div><label for="d-${k}">${k}</label><input id="d-${k}" name="${k}" type="number" step="${step}" min="${lim[k][0]}" max="${lim[k][1]}" value="${det[k]}"></div>`;
  app().innerHTML = `
    <h1>Your data</h1>
    <p class="lead">Analyse your own recording with the same steps. The current dataset is <b>${esc(S.data ? (S.data.simulated ? 'the simulated demo' : S.data.title) : 'the simulated demo')}</b>.</p>
    <div class="two">
      <form class="panel params" id="tif">
        <h3 style="margin-top:0">A TIFF stack <span class="badge py">Python</span></h3>
        ${isStatic() ? needsServer('Detecting particles in your own recording') : ''}
        <label for="file">Recording (.tif / .tiff, frames × height × width; for 4D stacks the first channel)</label>
        <input id="file" name="file" type="file" accept=".tif,.tiff" ${isStatic() ? 'disabled' : ''}>
        <fieldset><legend>Detector</legend><div class="row">${num('threshold_rel', 0.01)}${num('min_distance', 1)}</div>
          <div class="row">${num('max_sigma', 0.1)}</div>
          <p class="hint">max_sigma sets the fitting window (4 × max_sigma each side). The detector also has min_sigma and
            sigma_steps, which it stores but does not use, so they are not offered here.</p></fieldset>
        <fieldset><legend>Recording</legend><div class="row">
          <div><label for="d-max">Frames to analyse (from the start)</label><input id="d-max" name="max_frames" type="number" min="1" max="${info.max_frames}" value="100"></div>
          <div><label for="d-dt">Frame interval (s)</label><input id="d-dt" type="number" step="any" min="0.00001" placeholder="${info.defaults.diffusion.frame_interval} (default)"></div></div>
          <p class="hint">The pixel size is read from the file (OME or ImageJ metadata); if it has none, ${info.defaults.diffusion.pixel_size} µm
            (DiffusionAnalyzer's default) is used and the Motion page says so. Detection runs on this computer; large stacks take a while
            (about 0.1 s per 128×128 frame).</p></fieldset>
        <button type="submit" ${isStatic() ? 'disabled' : ''}>Detect particles</button> <span id="tifmsg" class="small"></span>
      </form>
      <form class="panel params" id="csvf">
        <h3 style="margin-top:0">A detections table <span class="badge js">in your browser</span></h3>
        <p class="small">A CSV with columns <code>frame, x, y</code> (pixels; optional <code>intensity, sigma, snr</code>), as the package's
          <code>DataWriter.write_particles_csv</code> writes, or detections from another program. Tracking and features run here;
          there is no image to show, so tracks are drawn on a blank field.</p>
        <label for="csv">Detections CSV</label><input id="csv" type="file" accept=".csv,text/csv">
        <button type="submit">Load detections</button> <span id="csvmsg" class="small"></span>
      </form>
    </div>
    <p><button type="button" class="alt" id="demo">Back to the demo recording</button></p>`;

  $('#tif').onsubmit = async e => {
    e.preventDefault();
    const f = $('#file').files[0], msg = $('#tifmsg'), btn = e.target.querySelector('button');
    if (!f) { msg.innerHTML = '<span class="err">Choose a file first.</span>'; return; }
    const fd = new FormData();
    fd.append('file', f);
    for (const k of ['threshold_rel', 'min_distance', 'max_sigma']) fd.append(k, $('#d-' + k).value);
    fd.append('max_frames', $('#d-max').value);
    btn.disabled = true; msg.innerHTML = '<span class="mute">Reading and detecting...</span>';
    try {
      const r = await api('/detect', { body: fd });
      const dt = $('#d-dt').value ? +$('#d-dt').value : null;
      setUpload(r, dt);
      location.hash = '#/recording';
    } catch (err) { msg.innerHTML = `<span class="err">${esc(err.message)}</span>`; } finally { btn.disabled = false; }
  };
  $('#csvf').onsubmit = async e => {
    e.preventDefault();
    const f = $('#csv').files[0], msg = $('#csvmsg');
    if (!f) { msg.innerHTML = '<span class="err">Choose a file first.</span>'; return; }
    try {
      setDetections(f.name, parseDetectionsCsv(await f.text()));
      location.hash = '#/tracks';
    } catch (err) { msg.innerHTML = `<span class="err">${esc(err.message)}</span>`; }
  };
  $('#demo').onclick = async () => { S.data = null; await loadDemo(); location.hash = '#/recording'; };
});
