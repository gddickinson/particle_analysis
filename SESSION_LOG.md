# SESSION_LOG

## 2026-11-11/12: web interface, static site, bug fixes (branch `web-interface`)

### Bug fixes (separate commits, each with regression tests)
- Test collection: the repo-root `__init__.py` imported missing modules, so `pytest tests` collected nothing;
  `tests/test_visualization.py` was a truncated paste (SyntaxError). Removed / replaced with offscreen viewer smoke
  tests plus the original TrackVisualizer error-handling test.
- Tracking: `max_gap=N` bridged only N-1 missing frames and `max_gap=0` never linked; a track whose single nearest
  particle was claimed by a closer track was cut even with another particle in range. Linking is now closest-pair
  first over all pairs within `max_distance`. (Changes tracks: crossing/nearby particles no longer split; default
  max_gap=2 now allows two missing frames.)
- Features: asymmetry returned complex128 (CSV "(1+0j)"); tracks with MSD 0 at some lag (revisiting a position) or
  repeated positions failed entirely (features None). Now real float; fit uses positive-MSD lags; fractal dimension
  uses the smallest non-zero distance.
- Statistics: velocity map empty bins were uninitialised memory (`np.divide(where=)` without `out=`).
- plot_utils.TrackVisualizer: `plt.cm.get_cmap` (removed in matplotlib 3.9) and `scatter(c=..., color=...)` made
  plot_tracks, plot_msd_curves and Export Movie fail.
- Packaging: scikit-learn, matplotlib, openpyxl were missing from pyproject/requirements. README corrected
  (single-scale detection, relative threshold, linking rule, requirements path, contributing link).
- Wrong test expectations fixed (asymmetry of a line, 3 clusters from 2 tracks, NaN == NaN).

### Method fixes (2026-11-12, at George's request: "fix ... any other problems")
- `fit_msd_curve`: D = exp(intercept)/4 for MSD = 4 D t^alpha (was /(4 alpha)); a negative slope is an invalid fit
  (D, alpha NaN).
- Fractal dimension: Katz's estimator (1 for a line, ~1.9 for 2D random walks, >2 confined) replaces box counting
  (~0.5 for a random walk).
- Motion classification: GLS fits weighted by the covariance of a Brownian track's MSD, model chosen by lowest BIC
  (was raw R^2). Simulated 100-frame tracks: ~87% diffusive and confined, 100% directed classified as simulated
  (OLS + AIC/BIC alone got diffusive only 25-43% right because the MSD lags are correlated).
- `MotionType.UNCLASSIFIED` for < 4 MSD points or failed fits (were reported as normal diffusion).
- `ParticleDetector`: min_sigma/max_sigma bound the fitted width; sigma_steps trial widths give the start value.
- Both former strict xfails pass (the confined test fixture now reaches its boundary within the analysed lags).

### Still open (not changed)
- `TrackAnalyzer.calculate_spatial_statistics` may mix axes for non-square frames (unused by the app).
- The GUI dashboard uses DiffusionAnalyzer's default pixel size / frame interval (0.108 um, 0.1 s) silently.
- `gui/main_window.py` (930 lines) and `gui/results_viewer.py` (508) exceed the 500-line budget.

### Web interface
- `src/particle_analysis/web/` (FastAPI, port 8129) + static site (GitHub Pages workflow). Pages: home, Detect,
  Track, Features, Motion, Your data, About. Demo recording simulated by the new `simulation.py` (labelled).
- Python: detection and motion analysis (precomputed for the demo: 5 thresholds). JS ports of ParticleTracker and
  FeatureCalculator run in the browser; `engine.test.mjs` checks 9 cases (275 tracks, 162 feature rows) against the
  Python output.
- Upload: TIFF stack -> Python detection (local app; pixel size from OME/ImageJ metadata, else default, stated);
  detections CSV -> tracking/features in the browser (also on the static site).
- The repo's history has `examples/test.tif` (deleted in 120c785; 4 frames, Hamamatsu camera): not used.
