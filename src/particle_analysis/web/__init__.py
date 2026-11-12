"""Web interface for ParticleAnalysis (FastAPI app + plain ES-module front end, no build step).

- app.py        routes under /api and STATIC_ENDPOINTS (exported by scripts/build_static.py for GitHub Pages)
- pipeline.py   the package's detector, tracker, feature calculator and diffusion analyzer wrapped for the API
- recording.py  reading an uploaded TIFF stack (DataReader) and its pixel size from the file's metadata
- demo.py       the simulated demo recording (particle_analysis.simulation) and its precomputed analysis
- static/       front end: pages, viewer, JS ports of the tracker and track features (+ node tests)
"""
