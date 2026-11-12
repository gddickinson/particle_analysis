# tests/test_visualization.py
"""Smoke tests of the viewers: Qt TrackViewer/FeatureViewer (visualization/viewers.py, offscreen) and the matplotlib
TrackVisualizer (visualization/plot_utils.py).

The previous version of this file was a truncated paste (a SyntaxError on line 3; its fixtures and imports were lost)
whose Qt tests needed pytest-qt and called widget attributes the viewers do not have. These tests use a plain offscreen
QApplication and are skipped when PyQt6/pyqtgraph are not installed; the TrackVisualizer error-handling test is the
original one, restored.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
try:  # the Qt stack can be absent or broken (e.g. two Qt bindings loaded); skip rather than error
    from PyQt6 import QtWidgets
    from particle_analysis.visualization.viewers import FeatureViewer, TrackViewer
except ImportError as e:
    pytest.skip(f"PyQt6/pyqtgraph not importable: {e}", allow_module_level=True)

from particle_analysis.core.feature_calculation import FeatureCalculator  # noqa: E402
from particle_analysis.core.particle_detection import Particle  # noqa: E402
from particle_analysis.core.particle_tracking import Track  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def sample_tracks():
    rng = np.random.default_rng(1)
    tracks = []
    for tid in range(3):
        xy = np.cumsum(rng.normal(0, 1, (12, 2)), axis=0) + 20 * tid
        particles = [Particle(frame=i, x=float(x), y=float(y), intensity=100.0, sigma=1.5, snr=10.0,
                              frame_size=(100, 100)) for i, (x, y) in enumerate(xy)]
        tracks.append(Track(id=tid, particles=particles, start_frame=0, end_frame=11))
    return tracks


def test_track_viewer_draws_tracks(qapp, sample_tracks):
    viewer = TrackViewer()
    viewer.set_data(sample_tracks)
    viewer.set_current_frame(11)
    assert set(viewer.track_plots) == {0, 1, 2}
    viewer.clear()
    assert not viewer.track_plots


def test_feature_viewer_lists_numeric_features(qapp, sample_tracks):
    features = [FeatureCalculator().calculate_track_features(t) for t in sample_tracks]
    viewer = FeatureViewer()
    viewer.set_features(features)
    names = [viewer.feature_combo.itemText(i) for i in range(viewer.feature_combo.count())]
    assert "diffusion_coefficient" in names and "msd_values" not in names
    assert "Mean:" in viewer.stats_label.text()


def test_visualization_error_handling(sample_tracks, tmp_path):
    """Test error handling in visualization (restored from the original file)"""
    import matplotlib
    matplotlib.use("Agg")
    from particle_analysis.visualization.plot_utils import TrackVisualizer
    visualizer = TrackVisualizer()

    # Test with empty track list
    assert visualizer.plot_tracks([]) is not None
    # Test with invalid feature names
    assert visualizer.plot_tracks(sample_tracks, color_by='invalid_feature') is not None
    # Test movie creation with invalid path
    assert not visualizer.create_track_movie(sample_tracks, str(tmp_path / "missing" / "movie.mp4"))
