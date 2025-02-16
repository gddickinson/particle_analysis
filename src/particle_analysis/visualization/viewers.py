# src/particle_analysis/visualization/viewers.py

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QComboBox, QLabel, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional, Union
from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures

class TrackViewer(QWidget):
    """Widget for interactive visualization of particle tracks"""

    # Signals
    track_selected = pyqtSignal(int)  # Emitted when track is selected
    frame_changed = pyqtSignal(int)   # Emitted when current frame changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []
        self.features = []
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setBackground('w')
        # Invert the y-axis to match image coordinates
        self.plot_widget.getViewBox().invertY(True)
        layout.addWidget(self.plot_widget)

        # Create control panel
        control_layout = QHBoxLayout()

        # Color control
        color_label = QLabel("Color by:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(['track_id', 'intensity', 'diffusion'])
        self.color_combo.currentTextChanged.connect(self.update_colors)

        # Frame control
        frame_label = QLabel("Frame:")
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.valueChanged.connect(self.set_current_frame)

        # Display controls
        self.show_points_btn = QPushButton("Show Points")
        self.show_points_btn.setCheckable(True)
        self.show_points_btn.setChecked(True)
        self.show_points_btn.toggled.connect(self.update_display)

        self.show_trails_btn = QPushButton("Show Trails")
        self.show_trails_btn.setCheckable(True)
        self.show_trails_btn.setChecked(True)
        self.show_trails_btn.toggled.connect(self.update_display)

        # Add controls to layout
        control_layout.addWidget(color_label)
        control_layout.addWidget(self.color_combo)
        control_layout.addWidget(frame_label)
        control_layout.addWidget(self.frame_spin)
        control_layout.addWidget(self.show_points_btn)
        control_layout.addWidget(self.show_trails_btn)
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Initialize plot items
        self.track_plots = {}  # Dictionary to store plot items for each track
        self.point_plots = {}  # Dictionary to store scatter plot items

    def set_data(self,
                 tracks: List[Track],
                 features: Optional[List[TrackFeatures]] = None):
        """
        Set tracks and features to display

        Parameters
        ----------
        tracks : List[Track]
            List of tracks to visualize
        features : List[TrackFeatures], optional
            Track features for coloring
        """
        self.tracks = tracks
        self.features = features if features is not None else []

        # Update frame range
        max_frame = max(t.end_frame for t in tracks)
        self.frame_spin.setMaximum(max_frame)

        # Update color options if features are provided
        if features:
            current_items = [self.color_combo.itemText(i)
                           for i in range(self.color_combo.count())]
            # Add new feature options
            feature_dict = features[0].to_dict()
            for name, value in feature_dict.items():
                if isinstance(value, (int, float)) and name not in current_items:
                    self.color_combo.addItem(name)

        # Initial plot update
        self.update_display()

    def set_current_frame(self, frame: int):
        """Set the current frame to display"""
        self.frame_spin.setValue(frame)
        self.frame_changed.emit(frame)
        self.update_display()

    def update_colors(self):
        """Update track colors based on selected property"""
        color_by = self.color_combo.currentText()

        if color_by == 'track_id':
            colors = [pg.mkColor(name) for name in pg.colormap.get('viridis').getLookupTable(nPts=len(self.tracks))]
        elif self.features and hasattr(self.features[0], color_by):
            values = [getattr(f, color_by) for f in self.features]
            norm_values = (np.array(values) - min(values)) / (max(values) - min(values))
            colors = [pg.mkColor(name) for name in pg.colormap.get('viridis').getLookupTable(nPts=len(values), pos=norm_values)]
        else:
            colors = [pg.mkColor('blue')] * len(self.tracks)

        # Update plot colors
        for track_id, plot_item in self.track_plots.items():
            color_idx = next(i for i, track in enumerate(self.tracks) if track.id == track_id)
            plot_item.setPen(colors[color_idx])

        for track_id, scatter_item in self.point_plots.items():
            color_idx = next(i for i, track in enumerate(self.tracks) if track.id == track_id)
            scatter_item.setBrush(colors[color_idx])

    def update_display(self):
        """Update the track display"""
        self.plot_widget.clear()
        current_frame = self.frame_spin.value()

        for track in self.tracks:
            # Get positions up to current frame
            mask = track.frames <= current_frame
            positions = track.positions[mask]

            if len(positions) > 0:
                # Plot trail if enabled
                if self.show_trails_btn.isChecked():
                    trail_item = pg.PlotDataItem(
                        positions[:, 0], positions[:, 1],
                        pen=pg.mkPen('b', width=2)
                    )
                    self.plot_widget.addItem(trail_item)
                    self.track_plots[track.id] = trail_item

                # Plot points if enabled
                if self.show_points_btn.isChecked():
                    scatter_item = pg.ScatterPlotItem(
                        positions[-1:, 0], positions[-1:, 1],
                        brush=pg.mkBrush('b'), size=10
                    )
                    self.plot_widget.addItem(scatter_item)
                    self.point_plots[track.id] = scatter_item

        # Update colors
        self.update_colors()

    def clear(self):
        """Clear all plots"""
        self.plot_widget.clear()
        self.track_plots.clear()
        self.point_plots.clear()

class FeatureViewer(QWidget):
    """Widget for visualizing track features"""

    feature_selected = pyqtSignal(str)  # Emitted when feature is selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.features = []
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Feature selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Feature:"))
        self.feature_combo = QComboBox()
        self.feature_combo.currentTextChanged.connect(self.update_display)
        selection_layout.addWidget(self.feature_combo)
        layout.addLayout(selection_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Count')
        layout.addWidget(self.plot_widget)

        # Add statistics panel
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

    def set_features(self, features: List[TrackFeatures]):
        """Set features to display"""
        self.features = features

        # Update feature options
        self.feature_combo.clear()
        if features:
            feature_dict = features[0].to_dict()
            # Add all numeric features except track_id and msd_values
            for name, value in feature_dict.items():
                if isinstance(value, (int, float)) and name != 'track_id' and name != 'msd_values':
                    self.feature_combo.addItem(name)

            # Set initial feature if none selected
            if self.feature_combo.currentText() == "":
                self.feature_combo.setCurrentIndex(0)

            self.update_display()

    def update_display(self):
        """Update the feature display"""
        self.plot_widget.clear()

        if not self.features:
            return

        feature_name = self.feature_combo.currentText()
        if not feature_name:
            return

        # Get feature values
        values = [getattr(f, feature_name) for f in self.features]

        # Create histogram
        y, x = np.histogram(values, bins='auto')
        bin_width = x[1] - x[0]

        # Create bar graph
        bargraph = pg.BarGraphItem(x=x[:-1], height=y, width=bin_width, brush='b')
        self.plot_widget.addItem(bargraph)

        # Update labels
        self.plot_widget.setLabel('bottom', feature_name.replace('_', ' ').title())
        self.plot_widget.setLabel('left', 'Count')

        # Update statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        stats_text = (
            f"Statistics for {feature_name.replace('_', ' ').title()}:\n"
            f"Mean: {mean_val:.3f}\n"
            f"Std Dev: {std_val:.3f}\n"
            f"Min: {min_val:.3f}\n"
            f"Max: {max_val:.3f}"
        )
        self.stats_label.setText(stats_text)

        # Emit signal
        self.feature_selected.emit(feature_name)

    def clear(self):
        """Clear the display"""
        self.features = []
        self.feature_combo.clear()
        self.plot_widget.clear()
        self.stats_label.clear()
