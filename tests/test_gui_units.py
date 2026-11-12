# tests/test_gui_units.py
"""The GUI dashboard and results viewer require explicit physical units for the motion analysis.

Regression: both built DiffusionAnalyzer() with its defaults (0.108 um/px, 0.1 s) silently, so motion types, MSD and
exported D values were in units nobody had checked. Offscreen; skipped when the Qt stack cannot be imported.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
try:
    from PyQt6 import QtWidgets
    from particle_analysis.gui.analysis_dashboard import AnalysisDashboard
    from particle_analysis.gui.results_viewer import ResultsViewer
except ImportError as e:
    pytest.skip(f"PyQt6/pyqtgraph not importable: {e}", allow_module_level=True)

from particle_analysis.core.feature_calculation import FeatureCalculator  # noqa: E402
from particle_analysis.core.particle_detection import Particle  # noqa: E402
from particle_analysis.core.particle_tracking import Track  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    tracks = []
    for tid in range(4):
        xy = np.cumsum(rng.normal(0, 1, (40, 2)), axis=0) + 30 * tid
        ps = [Particle(frame=i, x=float(x), y=float(y), intensity=100.0, sigma=1.5, snr=8.0, frame_size=(200, 200))
              for i, (x, y) in enumerate(xy)]
        tracks.append(Track(id=tid, particles=ps, start_frame=0, end_frame=39))
    return tracks, [FeatureCalculator().calculate_track_features(t) for t in tracks]


def test_dashboard_needs_units(qapp, data, tmp_path, monkeypatch):
    tracks, feats = data
    dash = AnalysisDashboard()
    dash.set_data(tracks, feats)
    assert dash.units.units() is None and dash.units.analyzer() is None
    assert "not set" in dash.units.pixel_spin.text() and "units not set" in dash.status_label.text()
    assert not dash.motion_plot.getPlotItem().listDataItems()
    monkeypatch.chdir(tmp_path)
    dash.export_results()
    assert "Not exported" in dash.status_label.text() and not list(tmp_path.glob("*.xlsx"))
    dash.units.set_units(0.16, 0.05, "test metadata")
    assert dash.units.units() == (0.16, 0.05)
    assert dash.units.analyzer().pixel_size == 0.16
    assert dash.motion_plot.getPlotItem().listDataItems()
    assert "0.16" in dash.motion_plot.getPlotItem().titleLabel.text


def test_dashboard_takes_units_from_set_data(qapp, data):
    tracks, feats = data
    dash = AnalysisDashboard()
    dash.set_data(tracks, feats, pixel_size=0.2, frame_interval=0.03, units_source="OME metadata")
    assert dash.units.units() == (0.2, 0.03) and "OME metadata" in dash.units.describe()


def test_results_viewer_needs_units(qapp, data, tmp_path):
    tracks, feats = data
    viewer = ResultsViewer()
    viewer.set_data(tracks, feats)
    viewer.current_track_id = 0
    viewer.update_msd_plot()
    assert "units not set" in viewer.fit_params_label.text()
    assert viewer.export_results(tmp_path)
    out = tmp_path / "analysis_results"
    assert (out / "tracks.csv").exists() and not (out / "diffusion_analysis.csv").exists()
    viewer.set_data(tracks, feats, pixel_size=0.1, frame_interval=0.05, units_source="entered in test")
    viewer.current_track_id = 0
    viewer.update_msd_plot()
    assert "D =" in viewer.fit_params_label.text()  # the 'Normal Diffusion' fit is now found and shown
    assert viewer.export_results(tmp_path)
    head = (out / "diffusion_analysis.csv").read_text().splitlines()
    assert head[0].endswith("pixel_size_um,frame_interval_s") and head[1].endswith(",0.1,0.05")


def test_results_viewer_saves_plots(qapp, data, tmp_path):
    """Regression: save_plots called GraphicsScene.save(), which pyqtgraph does not have, so it always failed."""
    tracks, feats = data
    viewer = ResultsViewer()
    viewer.set_data(tracks, feats, pixel_size=0.1, frame_interval=0.05)
    viewer.resize(800, 600)
    assert viewer.save_plots(tmp_path)
    assert len(list((tmp_path / "plots").glob("*.png"))) == 5
