# src/particle_analysis/gui/analysis_dashboard.py

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QPushButton, QLabel, QComboBox, QSpinBox,
                            QTableWidget, QTableWidgetItem, QCheckBox,
                            QGroupBox, QScrollArea, QFrame, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import QtGui
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import logging

from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures
from ..analysis.diffusion import DiffusionAnalyzer, MotionType
from ..analysis.statistics import TrackAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

class MetricCard(QFrame):
    """Widget for displaying a single metric"""

    def __init__(self, title: str, value: str, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setup_ui(title, value)

    def setup_ui(self, title: str, value: str):
        """Set up the card UI"""
        layout = QVBoxLayout(self)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.value_label)

    def update_value(self, value: str):
        """Update the displayed value"""
        self.value_label.setText(value)

class QualityMetrics(QGroupBox):
    """Widget for displaying quality control metrics"""

    def __init__(self, parent=None):
        super().__init__("Quality Control", parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the quality metrics UI"""
        layout = QGridLayout(self)

        # Track length distribution
        self.length_plot = pg.PlotWidget(title="Track Length Distribution")
        layout.addWidget(self.length_plot, 0, 0)

        # SNR distribution
        self.snr_plot = pg.PlotWidget(title="Signal-to-Noise Ratio")
        layout.addWidget(self.snr_plot, 0, 1)

        # Localization precision
        self.precision_plot = pg.PlotWidget(title="Localization Precision")
        layout.addWidget(self.precision_plot, 1, 0)

        # Tracking confidence
        self.confidence_plot = pg.PlotWidget(title="Tracking Confidence")
        layout.addWidget(self.confidence_plot, 1, 1)

    def update_metrics(self, tracks: List[Track], features: List[TrackFeatures]):
        """Update quality metrics displays"""
        # Track length distribution
        lengths = [len(t.particles) for t in tracks]
        y, x = np.histogram(lengths, bins='auto')
        self.length_plot.clear()
        self.length_plot.plot(x[:-1], y, stepMode="center", fillLevel=0,
                            brush=(0,0,255,150))

        # SNR distribution
        snr_values = [p.snr for t in tracks for p in t.particles]
        y, x = np.histogram(snr_values, bins='auto')
        self.snr_plot.clear()
        self.snr_plot.plot(x[:-1], y, stepMode="center", fillLevel=0,
                          brush=(0,255,0,150))

        # Localization precision
        precisions = [p.sigma for t in tracks for p in t.particles]
        y, x = np.histogram(precisions, bins='auto')
        self.precision_plot.clear()
        self.precision_plot.plot(x[:-1], y, stepMode="center", fillLevel=0,
                               brush=(255,0,0,150))

class AnalysisDashboard(QWidget):
    """Main analysis dashboard widget"""

    # Signals
    data_filtered = pyqtSignal(list, list)  # Emitted when data is filtered
    export_requested = pyqtSignal(Path)  # Emitted when export is requested

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []
        self.features = []
        self.conditions = {}
        self.setup_ui()

    def setup_ui(self):
        """Set up the dashboard UI"""
        layout = QVBoxLayout(self)

        # Create toolbar
        toolbar = QHBoxLayout()

        # Condition selection
        toolbar.addWidget(QLabel("Condition:"))
        self.condition_combo = QComboBox()
        self.condition_combo.currentTextChanged.connect(self.update_display)
        toolbar.addWidget(self.condition_combo)

        # Quality filters
        toolbar.addWidget(QLabel("Filters:"))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setPrefix("Min Length: ")
        self.min_length_spin.valueChanged.connect(self.apply_filters)
        toolbar.addWidget(self.min_length_spin)

        self.min_snr_spin = QSpinBox()
        self.min_snr_spin.setPrefix("Min SNR: ")
        self.min_snr_spin.valueChanged.connect(self.apply_filters)
        toolbar.addWidget(self.min_snr_spin)

        toolbar.addStretch()

        # Export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        toolbar.addWidget(export_btn)

        layout.addLayout(toolbar)

        # Create scrollable area for dashboard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Create main dashboard content
        dashboard = QWidget()
        dashboard_layout = QVBoxLayout(dashboard)

        # Summary metrics
        metrics_group = QHBoxLayout()
        self.n_tracks_card = MetricCard("Total Tracks", "0")
        metrics_group.addWidget(self.n_tracks_card)

        self.mean_length_card = MetricCard("Mean Track Length", "0")
        metrics_group.addWidget(self.mean_length_card)

        self.mean_diffusion_card = MetricCard("Mean Diffusion Coef", "0")
        metrics_group.addWidget(self.mean_diffusion_card)

        self.mean_snr_card = MetricCard("Mean SNR", "0")
        metrics_group.addWidget(self.mean_snr_card)

        dashboard_layout.addLayout(metrics_group)

        # Quality metrics
        self.quality_metrics = QualityMetrics()
        dashboard_layout.addWidget(self.quality_metrics)

        # Motion analysis
        motion_group = QGroupBox("Motion Analysis")
        motion_layout = QHBoxLayout(motion_group)

        # Motion type distribution
        self.motion_plot = pg.PlotWidget(title="Motion Types")
        motion_layout.addWidget(self.motion_plot)

        # MSD analysis overview
        self.msd_plot = pg.PlotWidget(title="MSD Analysis")
        motion_layout.addWidget(self.msd_plot)

        dashboard_layout.addWidget(motion_group)

        # Feature correlations
        correlations_group = QGroupBox("Feature Correlations")
        correlations_layout = QGridLayout(correlations_group)

        # Feature selection
        correlations_layout.addWidget(QLabel("X Feature:"), 0, 0)
        self.x_feature_combo = QComboBox()
        self.x_feature_combo.currentTextChanged.connect(self.update_correlation_plot)
        correlations_layout.addWidget(self.x_feature_combo, 0, 1)

        correlations_layout.addWidget(QLabel("Y Feature:"), 0, 2)
        self.y_feature_combo = QComboBox()
        self.y_feature_combo.currentTextChanged.connect(self.update_correlation_plot)
        correlations_layout.addWidget(self.y_feature_combo, 0, 3)

        # Correlation plot
        self.correlation_plot = pg.PlotWidget()
        correlations_layout.addWidget(self.correlation_plot, 1, 0, 1, 4)

        dashboard_layout.addWidget(correlations_group)

        # Set dashboard as scroll area widget
        scroll.setWidget(dashboard)

        # Create status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        layout.addLayout(status_layout)

    def set_data(self,
                 tracks: List[Track],
                 features: List[TrackFeatures],
                 condition: Optional[str] = None):
        """Set data to display"""
        if condition is None:
            condition = "Default"

        self.conditions[condition] = {
            'tracks': tracks,
            'features': features
        }

        if condition not in self.condition_combo.currentText():
            self.condition_combo.addItem(condition)

        self.update_display()

    def update_display(self):
        """Update all dashboard displays"""
        condition = self.condition_combo.currentText()
        if not condition or condition not in self.conditions:
            return

        tracks = self.conditions[condition]['tracks']
        features = self.conditions[condition]['features']

        # Update summary metrics
        self.n_tracks_card.update_value(str(len(tracks)))
        self.mean_length_card.update_value(
            f"{np.mean([len(t.particles) for t in tracks]):.1f}"
        )
        self.mean_diffusion_card.update_value(
            f"{np.mean([f.diffusion_coefficient for f in features]):.3f}"
        )
        self.mean_snr_card.update_value(
            f"{np.mean([p.snr for t in tracks for p in t.particles]):.1f}"
        )

        # Update quality metrics
        self.quality_metrics.update_metrics(tracks, features)

        # Update motion analysis
        self.update_motion_analysis(tracks, features)

        # Update feature options and correlation plot
        if features:
            feature_names = [name for name, value in features[0].to_dict().items()
                           if isinstance(value, (int, float)) and name != 'track_id']

            current_x = self.x_feature_combo.currentText()
            current_y = self.y_feature_combo.currentText()

            self.x_feature_combo.clear()
            self.y_feature_combo.clear()

            self.x_feature_combo.addItems(feature_names)
            self.y_feature_combo.addItems(feature_names)

            # Restore previous selection if valid
            if current_x in feature_names:
                self.x_feature_combo.setCurrentText(current_x)
            if current_y in feature_names:
                self.y_feature_combo.setCurrentText(current_y)

            self.update_correlation_plot()

    def update_motion_analysis(self,
                             tracks: List[Track],
                             features: List[TrackFeatures]):
        """Update motion analysis plots"""
        analyzer = DiffusionAnalyzer()

        # Motion type distribution
        motion_types = [analyzer.classify_motion(track) for track in tracks]
        type_counts = {t: motion_types.count(t) for t in MotionType}

        self.motion_plot.clear()
        x = np.arange(len(MotionType))
        y = [type_counts[t] for t in MotionType]
        self.motion_plot.plot(x, y, stepMode="center", fillLevel=0,
                            brush=(0,0,255,150))

        # Add motion type labels
        ticks = [[(i, t.value) for i, t in enumerate(MotionType)]]
        self.motion_plot.getAxis('bottom').setTicks(ticks)

        # MSD overview
        self.msd_plot.clear()
        for track, feature in zip(tracks[:10], features[:10]):  # Plot first 10 tracks
            time_lags, msd_values = analyzer.calculate_msd(track)
            self.msd_plot.plot(time_lags, msd_values, pen=(np.random.randint(255),
                                                          np.random.randint(255),
                                                          np.random.randint(255),
                                                          100))

    def update_correlation_plot(self):
        """Update feature correlation plot"""
        condition = self.condition_combo.currentText()
        if not condition or condition not in self.conditions:
            return

        features = self.conditions[condition]['features']
        if not features:
            return

        x_feature = self.x_feature_combo.currentText()
        y_feature = self.y_feature_combo.currentText()

        if not x_feature or not y_feature:
            return

        x_values = [getattr(f, x_feature) for f in features]
        y_values = [getattr(f, y_feature) for f in features]

        self.correlation_plot.clear()
        self.correlation_plot.plot(x_values, y_values, pen=None, symbol='o',
                                 symbolSize=5, symbolBrush=(0,0,255,150))

        self.correlation_plot.setLabel('bottom', x_feature)
        self.correlation_plot.setLabel('left', y_feature)

        # Calculate and display correlation coefficient
        correlation = np.corrcoef(x_values, y_values)[0,1]
        self.correlation_plot.setTitle(f"Correlation: {correlation:.3f}")


    def apply_filters(self):
        """Apply quality filters to data"""
        condition = self.condition_combo.currentText()
        if not condition or condition not in self.conditions:
            return

        tracks = self.conditions[condition]['tracks']
        features = self.conditions[condition]['features']

        # Apply length filter
        min_length = self.min_length_spin.value()
        length_mask = [len(t.particles) >= min_length for t in tracks]

        # Apply SNR filter
        min_snr = self.min_snr_spin.value()
        snr_mask = [np.mean([p.snr for p in t.particles]) >= min_snr
                   for t in tracks]

        # Combine filters
        combined_mask = [l and s for l, s in zip(length_mask, snr_mask)]

        filtered_tracks = [t for t, m in zip(tracks, combined_mask) if m]
        filtered_features = [f for f, m in zip(features, combined_mask) if m]

        # Update display with filtered data
        self.update_display()

        # Emit signal with filtered data
        self.data_filtered.emit(filtered_tracks, filtered_features)

        # Update status
        n_filtered = len(tracks) - len(filtered_tracks)
        self.status_label.setText(
            f"Filtered out {n_filtered} tracks ({n_filtered/len(tracks)*100:.1f}%)"
        )

    def export_results(self):
        """Export analysis results"""
        condition = self.condition_combo.currentText()
        if not condition or condition not in self.conditions:
            return

        tracks = self.conditions[condition]['tracks']
        features = self.conditions[condition]['features']

        # Create summary DataFrame
        track_data = []
        for track, feature in zip(tracks, features):
            track_dict = {
                'track_id': track.id,
                'n_particles': len(track.particles),
                'duration': track.end_frame - track.start_frame,
                'mean_snr': np.mean([p.snr for p in track.particles]),
                'mean_intensity': np.mean([p.intensity for p in track.particles])
            }

            # Add feature data
            track_dict.update(feature.to_dict())
            track_data.append(track_dict)

        summary_df = pd.DataFrame(track_data)

        # Add quality metrics
        quality_data = {
            'min_track_length': self.min_length_spin.value(),
            'min_snr': self.min_snr_spin.value(),
            'n_tracks_before_filtering': len(tracks),
            'n_tracks_after_filtering': len(track_data),
            'mean_track_length': summary_df['n_particles'].mean(),
            'mean_snr': summary_df['mean_snr'].mean()
        }
        quality_df = pd.DataFrame([quality_data])

        # Calculate motion type distribution
        analyzer = DiffusionAnalyzer()
        motion_types = [analyzer.classify_motion(track) for track in tracks]
        motion_dist = {t.value: motion_types.count(t) for t in MotionType}
        motion_df = pd.DataFrame([motion_dist])

        # Export to Excel
        path = Path(condition).with_suffix('.xlsx')
        with pd.ExcelWriter(path) as writer:
            summary_df.to_excel(writer, sheet_name='Track Summary', index=False)
            quality_df.to_excel(writer, sheet_name='Quality Metrics', index=False)
            motion_df.to_excel(writer, sheet_name='Motion Types', index=False)

        self.status_label.setText(f"Results exported to {path}")
        self.export_requested.emit(path)

    def show_progress(self, show: bool = True):
        """Show or hide progress bar"""
        self.progress_bar.setVisible(show)

    def update_progress(self, value: int, maximum: int = 100):
        """Update progress bar"""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def set_status(self, message: str):
        """Update status message"""
        self.status_label.setText(message)
