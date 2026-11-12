# tests/test_visualization.py
"""Offscreen smoke tests of the Qt viewers in visualization/viewers.py (TrackViewer, FeatureViewer).

The previous version of this file was a truncated paste (a SyntaxError on line 3) that tested a TrackVisualizer class
the package does not have and needed pytest-qt. These tests use a plain offscreen QApplication instead and are skipped
when PyQt6/pyqtgraph are not installed.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pyqtgraph")

from particle_analysis.core.feature_calculation import FeatureCalculator  # noqa: E402
from particle_analysis.core.particle_detection import Particle  # noqa: E402
from particle_analysis.core.particle_tracking import Track  # noqa: E402
from particle_analysis.visualization.viewers import FeatureViewer, TrackViewer  # noqa: E402


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
