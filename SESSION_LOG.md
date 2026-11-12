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

### Needs George's decision (kept as strict xfail tests where relevant)
- `FeatureCalculator.fit_msd_curve`: D = exp(intercept) / (4 alpha). For MSD = 4 D t^alpha the usual estimate is
  exp(intercept) / 4; dividing by alpha gives negative D for alpha < 0 (seen for immobile particles in the demo).
  `plot_utils.plot_msd_curves` draws 4 D t^alpha, inconsistent with this D. Not changed.
- `DiffusionAnalyzer.classify_motion` picks the highest raw R^2; the 2-parameter confined/anomalous models contain
  normal diffusion, so diffusive tracks are often called confined (demo: 1 of 5 diffusive called normal). AIC/BIC
  (ROADMAP) would change results. `test_motion_classification_normal_and_confined` is xfail.
- `classify_motion` returns NORMAL_DIFFUSION when the fits fail (tracks < 8 points); the web shows "not classified"
  instead, the GUI dashboard still counts them as normal diffusion.
- Fractal dimension by box counting over few points gives ~0.5 for a random walk, not ~2 (xfail test).
- `ParticleDetector`: `min_sigma` and `sigma_steps` are unused (detection is single-scale); fit sigma bounds 0.8-4 px
  are fixed. README now says so.
- `TrackAnalyzer.calculate_spatial_statistics` indexes the map as [x_bin, y_bin] with n_bins from frame_size[0]/[1];
  for non-square frames given as (rows, cols) this mixes axes. Unused by the app; not changed.
- Default `pixel_size=0.108 um`, `frame_interval=0.1 s` are used silently by the GUI dashboard's DiffusionAnalyzer().

### Web interface
- `src/particle_analysis/web/` (FastAPI, port 8129) + static site (GitHub Pages workflow). Pages: home, Detect,
  Track, Features, Motion, Your data, About. Demo recording simulated by the new `simulation.py` (labelled).
- Python: detection and motion analysis (precomputed for the demo: 5 thresholds). JS ports of ParticleTracker and
  FeatureCalculator run in the browser; `engine.test.mjs` checks 9 cases (275 tracks, 162 feature rows) against the
  Python output.
- Upload: TIFF stack -> Python detection (local app; pixel size from OME/ImageJ metadata, else default, stated);
  detections CSV -> tracking/features in the browser (also on the static site).
- The repo's history has `examples/test.tif` (deleted in 120c785; 4 frames, Hamamatsu camera): not used.
