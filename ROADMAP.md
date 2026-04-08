# ParticleAnalysis — Roadmap

## Current State
The most professionally structured project in the collection. Proper Python package layout under `src/particle_analysis/` with subpackages: `core/` (detection, tracking, feature_calculation), `analysis/` (statistics, diffusion), `gui/` (main_window, image_viewer, results_viewer, analysis_dashboard, batch_dialog, settings_dialog, workers, help_dialog), `io/` (readers, writers), `visualization/` (plot_utils, viewers). Has `pyproject.toml`, `setup.py`, `requirements/` (dev and prod), `docs/`, `examples/` with real data, and `tests/`. PyQt6-based GUI with background workers.

## Short-term Improvements
- [ ] Populate `tests/` with unit tests for `core/particle_detection.py` and `core/particle_tracking.py` using synthetic data
- [ ] Add CI configuration (GitHub Actions) running tests and linting on push
- [ ] Add progress reporting in `core/particle_detection.py` `detect_movie()` for large stacks
- [ ] Improve error messages when input TIFF files have unexpected dimensions or dtypes
- [ ] Add parameter presets for common microscopy setups (TIRF, confocal, widefield) in `gui/settings_dialog.py`
- [ ] Ensure `gui/workers.py` properly handles cancellation during long-running detection/tracking

## Feature Enhancements
- [ ] Add ensemble MSD fitting — fit multiple diffusion models and select best via BIC/AIC
- [ ] Add hidden Markov model trajectory segmentation for particles switching motion states
- [ ] Support 3D particle tracking (z-stack data) in `core/particle_tracking.py`
- [ ] Add colocalization analysis for multi-channel data
- [ ] Implement drift correction using fiducial markers or cross-correlation
- [ ] Add a "quick analysis" mode that runs detection + tracking + analysis with default parameters
- [ ] Export analysis reports as PDF with embedded figures

## Long-term Vision
- [ ] Add deep learning-based detection (U-Net or similar) as an alternative to Gaussian fitting
- [ ] Build a headless CLI mode for batch processing on HPC clusters
- [ ] Add a REST API for integration with OMERO and other LIMS systems
- [ ] Support real-time acquisition analysis (stream frames from microscope software)
- [ ] Publish to PyPI and conda-forge for community distribution
- [ ] Add a napari plugin version for users who prefer that ecosystem

## Technical Debt
- [ ] Review `initFile_creator.py`, `project-structure.py`, `setup-files.py` — these appear to be scaffolding scripts that should be removed or moved to a `scripts/` directory
- [ ] Consolidate `setup.py` and `pyproject.toml` — prefer `pyproject.toml` as the single source
- [ ] Add type hints to `analysis/diffusion.py` MSD fitting functions
- [ ] The `docs/` directory has many markdown files — set up Sphinx or MkDocs to build them into a site
- [ ] Verify that `examples/__init__.py` is needed — data directories typically should not be packages
- [ ] Profile `core/feature_calculation.py` for vectorization opportunities with numpy
