# INTERFACE.md: navigation map

Particle detection, tracking and diffusion analysis for fluorescence microscopy recordings. Python package in
`src/particle_analysis/` with a PyQt6 desktop app and a FastAPI web interface.

## Package `src/particle_analysis/`

| Path | Contents |
|---|---|
| `__main__.py` | `python -m particle_analysis`: logging, QApplication, `gui.main_window.MainWindow` |
| `core/particle_detection.py` | `Particle` (dataclass), `ParticleDetector` (median-filter background, `peak_local_max`, 2D Gaussian fit per peak with sigma in [min_sigma, max_sigma]; `detect_frame`, `detect_movie`) |
| `core/particle_tracking.py` | `Track` (dataclass: positions, frames, intensities), `ParticleTracker` (`track_particles`: closest-pair-first linking within `max_distance`, gaps up to `max_gap` filled by interpolation, `min_track_length`) |
| `core/feature_calculation.py` | `TrackFeatures`, `FeatureCalculator` (MSD, log-log fit of 4 D t^alpha, Rg, asymmetry, Katz fractal dimension, straightness, velocity/acceleration, confinement ratio; pixels and frames) |
| `analysis/diffusion.py` | `MotionType` (incl. UNCLASSIFIED), `msd_covariance` (Brownian MSD covariance), `DiffusionAnalyzer(pixel_size, frame_interval)`: MSD in um^2/s, normal/anomalous/confined/directed GLS fits with BIC, `select_model`, `classify_motion`, `analyze_track` |
| `analysis/statistics.py` | `TrackAnalyzer`: KMeans classification of features, condition comparison, mobility, spatial maps, Excel report |
| `io/readers.py` | `DataReader`: `read_movie` (TIFF via skimage), particles/tracks/features CSV |
| `io/writers.py` | `DataWriter`: particles/tracks/features CSV, analysis summary, batch results |
| `visualization/viewers.py` | Qt widgets `TrackViewer`, `FeatureViewer` (pyqtgraph) |
| `visualization/plot_utils.py` | `TrackVisualizer` (matplotlib: tracks, feature distributions, MSD curves, track movie) |
| `gui/` | `main_window.py` (MainWindow, AnalysisWorker), `workers.py` (Detection/Tracking QThreads), `image_viewer.py`, `results_viewer.py`, `analysis_dashboard.py`, `batch_dialog.py`, `settings_dialog.py`, `help_dialog.py` |
| `simulation.py` | `SimulationParams`, `simulate_recording`: simulated recordings with ground truth (diffusive/confined/directed/immobile, blinking, Poisson + read noise); the web demo |
| `web/` | web interface (below) |

Flow: `DataReader.read_movie` -> `ParticleDetector.detect_movie` -> `ParticleTracker.track_particles` ->
`FeatureCalculator.calculate_track_features` / `DiffusionAnalyzer.analyze_track` -> `DataWriter` / viewers.

## Web interface `src/particle_analysis/web/`

| Path | Contents |
|---|---|
| `app.py` | FastAPI app: `/api/info`, `/api/demo/movie`, `/api/demo/detections/{thr}`, `/api/demo/motion/{thr}`, `POST /api/detect` (TIFF upload), `POST /api/motion`; `STATIC_ENDPOINTS` |
| `pipeline.py` | `DEFAULTS` (from the classes' signatures), parameter parsing, `detect`, `track`, `features`, `motion`, JSON conversion, `display_frames` |
| `recording.py` | `read_upload` (DataReader on uploaded bytes), `pixel_size_from_tiff` (OME / ImageJ metadata) |
| `demo.py` | the simulated demo recording and its precomputed detections (5 thresholds) and motion analysis |
| `static/` | `index.html`, `core.js` + `style.css` (shared kit), `style_app.css`, `main.js` (router), `state.js` (dataset, memoised tracks/features), `viewer.js` (movie canvas + overlays), `charts.js` (SVG charts, sortable tables), pages `home.js`, `recording.js`, `tracks.js`, `featurespage.js`, `motion.js`, `data.js`, `about.js`; engines `tracker.js`, `features.js` (JS ports) + `engine.test.mjs`; `static_api.js` (static site) |

## Scripts and tests

| Path | Contents |
|---|---|
| `run_web.py` | starts the web app on 127.0.0.1:8129 |
| `scripts/build_static.py` | exports the static site (front end + JSON of `STATIC_ENDPOINTS`) |
| `scripts/screenshots.py` | Playwright smoke test / README screenshots (`--static` or `--base URL`) |
| `scripts/make_engine_fixture.py` | writes `tests/fixtures/engine_cases.json` (Python tracker + features output for the JS tests) |
| `tests/` | unit tests per module; `test_web.py` (API, static build, node engine test, fixture is current) |
| `.github/workflows/pages.yml` | builds and publishes the static site on push to main |
