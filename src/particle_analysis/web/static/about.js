// About: what the project is, how each step works and where it runs here, conventions, how to run, credits.
import { app, esc, route } from './core.js';
import { getInfo } from './state.js';

const kv = o => `<dl class="kv">${Object.entries(o).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;

route('about', async () => {
  const info = await getInfo();
  const D = info.defaults, sim = info.demo;
  app().innerHTML = `
    <h1>About ParticleAnalysis</h1>
    <p class="lead">An open-source Python application for detecting, tracking and analysing fluorescent particles in
      microscopy image sequences, with a PyQt6 desktop interface (README). This web interface runs the same Python code.</p>
    <h2>The pipeline</h2>
    <ol class="steps">
      <li><b>Detection</b> (<code>core/particle_detection.py</code>, ParticleDetector) <span class="badge py">Python</span>:
        background by a 20×20 median filter, local maxima of the background-subtracted frame above <i>threshold_rel</i> × its
        maximum and at least <i>min_distance</i> apart, then a 2D Gaussian fitted to a window around each peak; fits with RMS
        residual above half the amplitude are rejected. SNR is the fitted amplitude over the standard deviation of the
        background-subtracted frame.</li>
      <li><b>Tracking</b> (<code>core/particle_tracking.py</code>, ParticleTracker) <span class="badge js">JS port in the browser</span>:
        frame by frame, all (track, detection) pairs within <i>max_distance</i> are linked closest first; a track may miss up
        to <i>max_gap</i> frames (the gap is filled by linear interpolation) and is kept with at least <i>min_track_length</i> points.</li>
      <li><b>Track features</b> (<code>core/feature_calculation.py</code>, FeatureCalculator) <span class="badge js">JS port in the browser</span>:
        MSD over up to ${D.features.max_msd_points} lags and a straight-line fit of log MSD against log lag for MSD = 4Dt<sup>α</sup> (α the slope; a falling MSD is an invalid fit),
        radius of gyration, asymmetry, fractal dimension (Katz), straightness, mean velocity and acceleration,
        confinement ratio; in pixels and frames.</li>
      <li><b>Motion analysis</b> (<code>analysis/diffusion.py</code>, DiffusionAnalyzer) <span class="badge py">Python</span>:
        the MSD in µm² against lag in seconds, fitted with normal, anomalous, confined and directed models; the best R² gives the
        motion type.</li>
    </ol>
    <p>The JS ports are checked against the Python output on the demo detections and hand-made cases
      (<code>tests/fixtures/engine_cases.json</code>, node test <code>engine.test.mjs</code>): same tracks, features within 1e-7.</p>
    <h2>Defaults</h2>
    <div class="three"><div class="panel"><h3 style="margin-top:0">Detector</h3>${kv(D.detector)}</div>
      <div class="panel"><h3 style="margin-top:0">Tracker</h3>${kv(D.tracker)}</div>
      <div class="panel"><h3 style="margin-top:0">Features &amp; motion</h3>${kv({ ...D.features, 'pixel_size (µm)': D.diffusion.pixel_size, 'frame_interval (s)': D.diffusion.frame_interval })}</div></div>
    <h2>Conventions</h2>
    <p>Positions are in pixels with the origin at the centre of the first pixel (the fit reports pixel centres; ImageJ and
      ThunderSTORM put the origin at the pixel corner, half a pixel away). Physical units appear only in the motion analysis;
      its pixel size comes from the file's OME or ImageJ metadata when present, otherwise DiffusionAnalyzer's default
      (${D.diffusion.pixel_size} µm), and the page says which was used. The frame interval is not read from files: enter it.</p>
    <h2>The demo recording</h2>
    <p><span class="badge sim">simulated</span> ${esc(sim.description)} ${sim.n_frames} frames of ${sim.width}×${sim.height} px,
      pixel size ${sim.pixel_size_um} µm, frame interval ${sim.frame_interval_s} s
      (<code>particle_analysis/simulation.py</code>, <code>SimulationParams()</code>). The repository has no microscope recording
      of its own, so the demo is simulated; its true particle paths let the Detect, Track and Motion pages check the results.</p>
    <h2>Run it</h2>
    <h3>Desktop app (PyQt6)</h3>
    <pre>pip install -r requirements/requirements.txt
pip install .
python -m particle_analysis</pre>
    <h3>Web app (this interface, on your own files)</h3>
    <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8129</pre>
    <p>The static site (GitHub Pages) shows the demo with detection and motion analysis precomputed; tracking, features and
      detections tables you load run in your browser. Detecting particles in your own TIFF stacks needs the local web app or
      the desktop app.</p>
    <h2>Credits</h2>
    <p>By George Dickinson, MIT licence. From the README's acknowledgments: the scikit-image team for image processing
      algorithms, the trackpy project for inspiration on particle tracking approaches, and the PyQt team for the GUI framework.
      Source: <a href="https://github.com/gddickinson/particle_analysis">github.com/gddickinson/particle_analysis</a>.</p>`;
});
