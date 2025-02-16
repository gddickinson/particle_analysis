# src/particle_analysis/gui/results_viewer.py

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                            QPushButton, QComboBox, QLabel, QSpinBox,
                            QSplitter, QTreeWidget, QTreeWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging

from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures
from ..visualization.viewers import TrackViewer, FeatureViewer
from ..analysis.diffusion import DiffusionAnalyzer, MotionType

# Set up logging
logger = logging.getLogger(__name__)

class ResultsViewer(QWidget):
    """Widget for displaying analysis results"""

    # Signals
    track_selected = pyqtSignal(int)  # Emitted when track is selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []
        self.features = []
        self.current_track_id = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        layout = QHBoxLayout(self)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Navigation tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Results"])
        self.tree.itemClicked.connect(self.handle_tree_selection)
        splitter.addWidget(self.tree)

        # Right panel - Results display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tabs for different views
        self.tabs = QTabWidget()

        # Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)

        # Add summary statistics
        stats_group = QWidget()
        stats_layout = QHBoxLayout(stats_group)
        self.n_tracks_label = QLabel("Number of Tracks: 0")
        self.mean_track_length_label = QLabel("Mean Track Length: 0")
        self.mean_diffusion_label = QLabel("Mean Diffusion Coef: 0")
        stats_layout.addWidget(self.n_tracks_label)
        stats_layout.addWidget(self.mean_track_length_label)
        stats_layout.addWidget(self.mean_diffusion_label)
        overview_layout.addWidget(stats_group)

        # Add track viewer to overview
        self.track_viewer = TrackViewer()
        overview_layout.addWidget(self.track_viewer)

        self.tabs.addTab(overview_tab, "Overview")

        # Track Details tab
        track_tab = QWidget()
        track_layout = QVBoxLayout(track_tab)

        # Track selection
        track_select_layout = QHBoxLayout()
        track_select_layout.addWidget(QLabel("Track:"))
        self.track_combo = QComboBox()
        self.track_combo.currentIndexChanged.connect(self.handle_track_selection)
        track_select_layout.addWidget(self.track_combo)
        track_layout.addLayout(track_select_layout)

        # Track visualization and details
        self.single_track_viewer = TrackViewer()
        track_layout.addWidget(self.single_track_viewer)

        # Track metrics
        metrics_group = QWidget()
        metrics_layout = QHBoxLayout(metrics_group)
        self.track_length_label = QLabel("Length: 0")
        self.track_duration_label = QLabel("Duration: 0")
        self.track_diffusion_label = QLabel("Diffusion Coef: 0")
        metrics_layout.addWidget(self.track_length_label)
        metrics_layout.addWidget(self.track_duration_label)
        metrics_layout.addWidget(self.track_diffusion_label)
        track_layout.addWidget(metrics_group)

        self.tabs.addTab(track_tab, "Track Details")

        # MSD Analysis tab
        msd_tab = QWidget()
        msd_layout = QVBoxLayout(msd_tab)

        # MSD plot
        self.msd_plot = pg.PlotWidget()
        self.msd_plot.setLabel('left', 'MSD')
        self.msd_plot.setLabel('bottom', 'Time Lag')
        msd_layout.addWidget(self.msd_plot)

        # Diffusion model controls
        model_group = QWidget()
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Normal Diffusion",
            "Anomalous Diffusion",
            "Confined Motion",
            "Directed Motion"
        ])
        self.model_combo.currentTextChanged.connect(self.update_msd_plot)
        model_layout.addWidget(self.model_combo)
        msd_layout.addWidget(model_group)

        # Model fit parameters
        self.fit_params_label = QLabel("Fit Parameters:")
        msd_layout.addWidget(self.fit_params_label)

        self.tabs.addTab(msd_tab, "MSD Analysis")

        # Feature Analysis tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)

        # Feature selection
        feature_select_layout = QHBoxLayout()
        feature_select_layout.addWidget(QLabel("Feature:"))
        self.feature_combo = QComboBox()
        self.feature_combo.currentTextChanged.connect(self.update_feature_plot)
        feature_select_layout.addWidget(self.feature_combo)
        feature_layout.addLayout(feature_select_layout)

        # Feature plot
        self.feature_plot = pg.PlotWidget()
        feature_layout.addWidget(self.feature_plot)

        # Feature statistics
        self.feature_stats_label = QLabel("Statistics:")
        feature_layout.addWidget(self.feature_stats_label)

        self.tabs.addTab(feature_tab, "Feature Analysis")

        # Diffusion Analysis tab
        diffusion_tab = QWidget()
        diffusion_layout = QVBoxLayout(diffusion_tab)

        # Motion type distribution
        self.motion_plot = pg.PlotWidget()
        self.motion_plot.setLabel('left', 'Count')
        self.motion_plot.setLabel('bottom', 'Motion Type')
        diffusion_layout.addWidget(self.motion_plot)

        # Diffusion coefficient distribution
        self.diffusion_plot = pg.PlotWidget()
        self.diffusion_plot.setLabel('left', 'Count')
        self.diffusion_plot.setLabel('bottom', 'Diffusion Coefficient')
        diffusion_layout.addWidget(self.diffusion_plot)

        self.tabs.addTab(diffusion_tab, "Diffusion Analysis")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([200, 800])

    def set_data(self,
                 tracks: List[Track],
                 features: List[TrackFeatures],
                 movie_data: Optional[np.ndarray] = None):
        """Set data to display"""
        self.tracks = tracks
        self.features = features
        self.movie_data = movie_data

        # Update navigation tree
        self.update_tree()

        # Update track selection
        self.track_combo.clear()
        self.track_combo.addItems([f"Track {t.id}" for t in tracks])

        # Update feature selection
        if features:
            feature_names = [name for name, value in features[0].to_dict().items()
                           if isinstance(value, (int, float)) and name != 'track_id']
            self.feature_combo.clear()
            self.feature_combo.addItems(feature_names)

        # Update overview
        self.update_overview()

        # Update current track if one is selected
        if self.current_track_id is not None:
            self.update_track_details()

    def update_tree(self):
        """Update navigation tree"""
        self.tree.clear()

        # Add overview item
        overview_item = QTreeWidgetItem(["Overview"])
        self.tree.addTopLevelItem(overview_item)

        # Add tracks
        tracks_item = QTreeWidgetItem(["Tracks"])
        self.tree.addTopLevelItem(tracks_item)
        for track in self.tracks:
            track_item = QTreeWidgetItem([f"Track {track.id}"])
            tracks_item.addChild(track_item)

        # Add analyses
        analysis_item = QTreeWidgetItem(["Analysis"])
        self.tree.addTopLevelItem(analysis_item)
        analysis_item.addChildren([
            QTreeWidgetItem(["MSD Analysis"]),
            QTreeWidgetItem(["Feature Analysis"]),
            QTreeWidgetItem(["Diffusion Analysis"])
        ])

    def handle_tree_selection(self, item: QTreeWidgetItem, column: int):
        """Handle navigation tree selection"""
        text = item.text(0)

        if text == "Overview":
            self.tabs.setCurrentIndex(0)
        elif text.startswith("Track"):
            self.tabs.setCurrentIndex(1)
            track_id = int(text.split()[1])
            self.track_combo.setCurrentIndex(
                [t.id for t in self.tracks].index(track_id)
            )
        elif text == "MSD Analysis":
            self.tabs.setCurrentIndex(2)
        elif text == "Feature Analysis":
            self.tabs.setCurrentIndex(3)
        elif text == "Diffusion Analysis":
            self.tabs.setCurrentIndex(4)

    def handle_track_selection(self, index: int):
        """Handle track selection"""
        if index >= 0:
            self.current_track_id = self.tracks[index].id
            self.update_track_details()
            self.track_selected.emit(self.current_track_id)

    def update_overview(self):
        """Update overview tab"""
        # Update summary statistics
        self.n_tracks_label.setText(f"Number of Tracks: {len(self.tracks)}")
        self.mean_track_length_label.setText(
            f"Mean Track Length: {np.mean([len(t.particles) for t in self.tracks]):.1f}"
        )
        if self.features:
            self.mean_diffusion_label.setText(
                f"Mean Diffusion Coef: {np.mean([f.diffusion_coefficient for f in self.features]):.3f}"
            )

        # Update track viewer
        self.track_viewer.set_data(self.tracks, self.features)

    def update_track_details(self):
        """Update track details tab"""
        if self.current_track_id is None:
            return

        track = next(t for t in self.tracks if t.id == self.current_track_id)
        feature = next(f for f in self.features if f.track_id == self.current_track_id)

        # Update track metrics
        self.track_length_label.setText(f"Length: {len(track.particles)}")
        duration = (track.end_frame - track.start_frame) * 0.1  # Assuming 0.1s interval
        self.track_duration_label.setText(f"Duration: {duration:.1f}s")
        self.track_diffusion_label.setText(
            f"Diffusion Coef: {feature.diffusion_coefficient:.3f}"
        )

        # Update track viewer
        self.single_track_viewer.set_data([track], [feature])

        # Update MSD plot
        self.update_msd_plot()

    def update_msd_plot(self):
        """Update MSD analysis plot"""
        if self.current_track_id is None:
            return

        track = next(t for t in self.tracks if t.id == self.current_track_id)

        # Calculate MSD
        analyzer = DiffusionAnalyzer()
        time_lags, msd_values = analyzer.calculate_msd(track)

        # Fit selected model
        fit_results = analyzer.fit_diffusion_models(time_lags, msd_values)
        model_name = self.model_combo.currentText().lower().replace(" ", "_")

        # Plot data and fit
        self.msd_plot.clear()
        self.msd_plot.plot(time_lags, msd_values, symbol='o')

        if model_name in fit_results:
            self.msd_plot.plot(time_lags, fit_results[model_name]['fit_values'],
                             pen='r')

            # Update fit parameters
            params = fit_results[model_name]
            param_text = f"R² = {params['r_squared']:.3f}\n"
            for key, value in params.items():
                if key not in ['fit_values', 'r_squared']:
                    param_text += f"{key} = {value:.3f}\n"
            self.fit_params_label.setText(param_text)

    def update_feature_plot(self):
        """Update feature analysis plot"""
        if not self.features:
            return

        feature_name = self.feature_combo.currentText()
        values = [getattr(f, feature_name) for f in self.features]

        # Create histogram
        self.feature_plot.clear()
        y, x = np.histogram(values, bins='auto')
        self.feature_plot.plot(x[:-1], y, stepMode="center", fillLevel=0,
                             brush=(0,0,255,150))

        # Update statistics
        stats_text = (
            f"Mean: {np.mean(values):.3f}\n"
            f"Std: {np.std(values):.3f}\n"
            f"Min: {np.min(values):.3f}\n"
            f"Max: {np.max(values):.3f}"
        )
        self.feature_stats_label.setText(stats_text)


    def update_diffusion_analysis(self):
        """Update diffusion analysis plots"""
        if not self.tracks or not self.features:
            return

        analyzer = DiffusionAnalyzer()

        # Motion type distribution
        motion_types = [analyzer.classify_motion(track) for track in self.tracks]
        type_counts = {t: motion_types.count(t) for t in MotionType}

        # Plot motion type distribution
        self.motion_plot.clear()
        x = np.arange(len(MotionType))
        y = [type_counts[t] for t in MotionType]
        self.motion_plot.plot(x, y, stepMode="center", fillLevel=0,
                            brush=(0,0,255,150))
        ticks = [[(i, t.value) for i, t in enumerate(MotionType)]]
        self.motion_plot.getAxis('bottom').setTicks(ticks)

        # Diffusion coefficient distribution
        diff_coefs = [f.diffusion_coefficient for f in self.features]

        self.diffusion_plot.clear()
        y, x = np.histogram(diff_coefs, bins='auto')
        self.diffusion_plot.plot(x[:-1], y, stepMode="center", fillLevel=0,
                               brush=(0,255,0,150))

    def export_results(self, directory: Path):
        """Export analysis results"""
        try:
            # Create results directory
            results_dir = directory / "analysis_results"
            results_dir.mkdir(exist_ok=True)

            # Export tracks
            with open(results_dir / "tracks.csv", 'w') as f:
                f.write("track_id,frame,x,y,intensity\n")
                for track in self.tracks:
                    for particle in track.particles:
                        f.write(f"{track.id},{particle.frame},"
                               f"{particle.x},{particle.y},"
                               f"{particle.intensity}\n")

            # Export features
            with open(results_dir / "features.csv", 'w') as f:
                if self.features:
                    # Write header
                    feature_dict = self.features[0].to_dict()
                    headers = list(feature_dict.keys())
                    f.write(",".join(headers) + "\n")

                    # Write data
                    for feature in self.features:
                        feature_dict = feature.to_dict()
                        values = [str(feature_dict[h]) for h in headers]
                        f.write(",".join(values) + "\n")

            # Export MSD analysis
            if self.current_track_id is not None:
                track = next(t for t in self.tracks
                           if t.id == self.current_track_id)
                analyzer = DiffusionAnalyzer()
                time_lags, msd_values = analyzer.calculate_msd(track)

                with open(results_dir / f"msd_track_{track.id}.csv", 'w') as f:
                    f.write("time_lag,msd\n")
                    for t, m in zip(time_lags, msd_values):
                        f.write(f"{t},{m}\n")

            # Export diffusion analysis
            analyzer = DiffusionAnalyzer()
            with open(results_dir / "diffusion_analysis.csv", 'w') as f:
                f.write("track_id,motion_type,diffusion_coefficient\n")
                for track, feature in zip(self.tracks, self.features):
                    motion_type = analyzer.classify_motion(track)
                    f.write(f"{track.id},{motion_type.value},"
                           f"{feature.diffusion_coefficient}\n")

            return True

        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return False

    def save_plots(self, directory: Path):
        """Save current plots"""
        try:
            # Create plots directory
            plots_dir = directory / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Save track overview
            self.track_viewer.plot_widget.scene().save(
                str(plots_dir / "track_overview.png"))

            # Save MSD plot
            self.msd_plot.scene().save(
                str(plots_dir / "msd_analysis.png"))

            # Save feature plot
            self.feature_plot.scene().save(
                str(plots_dir / "feature_analysis.png"))

            # Save diffusion plots
            self.motion_plot.scene().save(
                str(plots_dir / "motion_types.png"))
            self.diffusion_plot.scene().save(
                str(plots_dir / "diffusion_coefficients.png"))

            return True

        except Exception as e:
            logger.error(f"Error saving plots: {str(e)}")
            return False

    def copy_to_clipboard(self):
        """Copy current plot to clipboard"""
        current_tab = self.tabs.currentWidget()

        if hasattr(current_tab, 'plot_widget'):
            current_tab.plot_widget.scene().save_to_clipboard()

    def clear(self):
        """Clear all data and plots"""
        self.tracks = []
        self.features = []
        self.current_track_id = None

        # Clear track viewer
        self.track_viewer.clear()
        self.single_track_viewer.clear()

        # Clear plots
        self.msd_plot.clear()
        self.feature_plot.clear()
        self.motion_plot.clear()
        self.diffusion_plot.clear()

        # Clear selections
        self.track_combo.clear()
        self.feature_combo.clear()

        # Clear tree
        self.tree.clear()

        # Reset labels
        self.n_tracks_label.setText("Number of Tracks: 0")
        self.mean_track_length_label.setText("Mean Track Length: 0")
        self.mean_diffusion_label.setText("Mean Diffusion Coef: 0")
        self.track_length_label.setText("Length: 0")
        self.track_duration_label.setText("Duration: 0")
        self.track_diffusion_label.setText("Diffusion Coef: 0")
        self.fit_params_label.setText("Fit Parameters:")
        self.feature_stats_label.setText("Statistics:")
