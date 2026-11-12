// Home: hero, real counts from the demo recording and its precomputed detections, and cards to the pages.
import { api, app, esc, isStatic, route } from './core.js';
import { getInfo } from './state.js';

route('home', async () => {
  const info = await getInfo();
  const d = info.demo, det = await api('/demo/detections/' + d.default_threshold);
  const c = d.counts;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Single-particle tracking · fluorescence microscopy</p>
    <h1 class="glow">ParticleAnalysis</h1>
    <p class="lead">Find fluorescent particles in every frame of a recording by Gaussian fitting, link them into
      trajectories with nearest-neighbour matching and gap closing, then measure each track: mean square displacement,
      diffusion coefficient, anomalous exponent, shape descriptors and a motion type.</p>
    <a class="btn" href="#/recording">Open the demo recording</a><a class="btn alt" href="#/data">Analyse your data</a><a class="btn alt" href="#/about">How it works</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${d.n_frames}</b><span>frames in the demo (${d.width}×${d.height} px)</span></div>
    <div class="tile"><b>${d.n_particles}</b><span>simulated particles with known paths</span></div>
    <div class="tile"><b>${det.n.toLocaleString()}</b><span>detections at the default threshold (${esc(d.default_threshold)})</span></div>
    <div class="tile"><b>${info.defaults.tracker.max_distance} px</b><span>default linking distance</span></div>
    <div class="tile"><b>${d.pixel_size_um} µm</b><span>pixel size of the demo</span></div>
  </div>
  <div class="callout"><span class="badge sim">simulated</span> The demo recording is <b>simulated</b> by the repo's
    <code>particle_analysis.simulation</code> (${c.diffusive} diffusive, ${c.confined} confined, ${c.directed} directed and
    ${c.immobile} immobile particles, with blinking and camera noise), so every result can be checked against the true paths.
    ${isStatic() ? `On this static site detection and the motion analysis were run by the Python code when the site was built;
    tracking and track features run in your browser (JS ports tested against the Python). Your own TIFF recordings need the
    local app; a table of detections can be tracked here.` : 'This is the local app: your own recordings are analysed in Python on this computer.'}</div>
  <h2>Pipeline</h2>
  <div class="grid">
    <a class="card" href="#/recording"><b>1 · Detect</b><span>Median-filter background, local maxima above a relative threshold, a 2D Gaussian fit to each: sub-pixel position, amplitude, width, SNR.</span><small><span class="badge py">Python</span></small></a>
    <a class="card" href="#/tracks"><b>2 · Track</b><span>Link detections frame to frame (closest pairs first, within a maximum distance), bridge gaps, keep tracks of a minimum length.</span><small><span class="badge js">in your browser</span></small></a>
    <a class="card" href="#/features"><b>3 · Features</b><span>MSD and its log-log fit (D, α), radius of gyration, asymmetry, fractal dimension, straightness, velocity, confinement ratio.</span><small><span class="badge js">in your browser</span></small></a>
    <a class="card" href="#/motion"><b>4 · Motion</b><span>Fit normal, anomalous, confined and directed diffusion models to each MSD curve in µm² and seconds and classify the motion.</span><small><span class="badge py">Python</span></small></a>
    <a class="card" href="#/data"><b>Your data</b><span>Upload a TIFF stack (local app) or a detections table (anywhere) and run the same steps.</span></a>
    <a class="card" href="#/about"><b>About</b><span>Method details, conventions, how to run the desktop and web apps, credits.</span></a>
  </div>`;
});
