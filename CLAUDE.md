# ParticleAnalysis: notes for Claude

@INTERFACE.md

- Package under `src/particle_analysis/` (tests add `src/` to sys.path in `tests/conftest.py`). Run tests with
  `python -m pytest -q tests` (Qt tests run offscreen and are skipped without PyQt6).
- Web interface: `src/particle_analysis/web/` (FastAPI, port 8129, `python run_web.py`); static site built by
  `scripts/build_static.py`. The JS ports `web/static/tracker.js` and `features.js` must match the Python tracker and
  FeatureCalculator: after changing either, rerun `python scripts/make_engine_fixture.py` and
  `node src/particle_analysis/web/static/engine.test.mjs`.
- Coordinates are pixel centres; physical units only in `analysis/diffusion.py` (pixel_size, frame_interval).
- Keep files under 500 lines; update INTERFACE.md and SESSION_LOG.md with structural changes.
