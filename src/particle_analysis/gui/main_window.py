# src/particle_analysis/gui/main_window.py

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QFileDialog, QTabWidget, QLabel,
                            QProgressBar, QMessageBox, QSpinBox, QComboBox,
                            QCheckBox, QGroupBox, QToolBar, QStatusBar, QDoubleSpinBox, QSplitter, QTableWidget,
                            QTableWidgetItem, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph as pg
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

from ..core.particle_detection import ParticleDetector
from ..core.particle_tracking import ParticleTracker
from ..core.feature_calculation import FeatureCalculator
from ..io.readers import DataReader
from ..io.writers import DataWriter
from ..visualization.viewers import TrackViewer, FeatureViewer
from ..gui.image_viewer import ImageViewer
from ..gui.workers import DetectionWorker, TrackingWorker
from ..gui.settings_dialog import SettingsDialog
from ..gui.help_dialog import HelpDialog

# Set up logging
logger = logging.getLogger(__name__)

class AnalysisWorker(QThread):
    """Worker thread for running analysis"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path: str, parameters: dict):
        super().__init__()
        self.file_path = file_path
        self.parameters = parameters
        self.should_stop = False

    def run(self):
        try:
            # Load data
            reader = DataReader()
            movie = reader.read_movie(self.file_path)
            if movie is None:
                raise ValueError(f"Could not read movie file: {self.file_path}")

            self.progress.emit(10)
            if self.should_stop:
                return

            # Detect particles
            detector = ParticleDetector(
                min_sigma=self.parameters.get('min_sigma', 1.0),
                max_sigma=self.parameters.get('max_sigma', 3.0),
                threshold_rel=self.parameters.get('threshold_rel', 0.2)
            )
            particles = detector.detect_movie(movie)

            self.progress.emit(40)
            if self.should_stop:
                return

            # Track particles
            tracker = ParticleTracker(
                max_distance=self.parameters.get('max_distance', 5.0),
                max_gap=self.parameters.get('max_gap', 2),
                min_track_length=self.parameters.get('min_track_length', 3)
            )
            tracks = tracker.track_particles(particles)

            self.progress.emit(70)
            if self.should_stop:
                return

            # Calculate features
            calculator = FeatureCalculator()
            features = [calculator.calculate_track_features(track)
                       for track in tracks]
            features = [f for f in features if f is not None]

            self.progress.emit(90)
            if self.should_stop:
                return

            # Save results
            writer = DataWriter()
            base_path = Path(self.file_path).with_suffix('')
            writer.write_tracks_csv(tracks, f"{base_path}_tracks.csv")
            writer.write_features_csv(features, f"{base_path}_features.csv")

            self.progress.emit(100)

            # Return results
            results = {
                'particles': particles,
                'tracks': tracks,
                'features': features
            }
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Analysis error: {str(e)}")

    def stop(self):
        """Stop the analysis"""
        self.should_stop = True

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        """Initialize MainWindow"""
        super().__init__()
        self.setWindowTitle("Particle Analysis")

        # Initialize state
        self.current_results = None
        self.analysis_worker = None

        # Set up the UI first
        self.setup_ui()

        # Connect basic UI actions
        self.connect_actions()

        # Then set up menu bar (which will connect its own actions)
        self.setup_menubar()

        # Set window size
        self.resize(1200, 800)

    def setup_ui(self):
        """Set up the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Add file operations
        open_action = toolbar.addAction("Open File")
        open_action.triggered.connect(self.open_file)

        batch_action = toolbar.addAction("Batch Process")
        batch_action.triggered.connect(self.batch_process)

        toolbar.addSeparator()

        # Add analysis controls
        detect_action = toolbar.addAction("Detect Particles")
        detect_action.triggered.connect(self.run_detection)
        detect_action.setEnabled(False)
        self.detect_action = detect_action

        track_action = toolbar.addAction("Track Particles")
        track_action.triggered.connect(self.run_tracking)
        track_action.setEnabled(False)
        self.track_action = track_action

        run_action = toolbar.addAction("Analyze Full Image")
        run_action.triggered.connect(self.run_analysis)
        run_action.setEnabled(False)
        run_action.setToolTip("Detect and track particles in the entire image")
        self.run_action = run_action

        stop_action = toolbar.addAction("Stop")
        stop_action.triggered.connect(self.stop_analysis)
        stop_action.setEnabled(False)
        self.stop_action = stop_action

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Image and parameter controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Add parameter controls
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QVBoxLayout()

        # Detection parameters
        det_layout = QHBoxLayout()
        det_layout.addWidget(QLabel("Min Sigma:"))
        self.min_sigma_spin = QSpinBox()
        self.min_sigma_spin.setValue(1)
        det_layout.addWidget(self.min_sigma_spin)

        det_layout.addWidget(QLabel("Max Sigma:"))
        self.max_sigma_spin = QSpinBox()
        self.max_sigma_spin.setValue(3)
        det_layout.addWidget(self.max_sigma_spin)

        det_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setValue(0.2)
        det_layout.addWidget(self.threshold_spin)

        param_layout.addLayout(det_layout)

        # Tracking parameters
        track_layout = QHBoxLayout()
        track_layout.addWidget(QLabel("Max Distance:"))
        self.max_distance_spin = QSpinBox()
        self.max_distance_spin.setValue(5)
        track_layout.addWidget(self.max_distance_spin)

        track_layout.addWidget(QLabel("Max Gap:"))
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setValue(2)
        track_layout.addWidget(self.max_gap_spin)

        track_layout.addWidget(QLabel("Min Length:"))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setValue(3)
        track_layout.addWidget(self.min_length_spin)

        param_layout.addLayout(track_layout)
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # Add image viewer
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer)

        splitter.addWidget(left_panel)

        # Right panel - Analysis results
        # Create tabs for different views
        tabs = QTabWidget()

        # Track visualization tab
        self.track_viewer = TrackViewer()
        tabs.addTab(self.track_viewer, "Visualization")

        # Feature analysis tab
        self.feature_viewer = FeatureViewer()
        tabs.addTab(self.feature_viewer, "Analysis")

        # Add new results tables tab
        tables_tab = self.setup_results_tables()
        tabs.addTab(tables_tab, "Results Tables")

        splitter.addWidget(tabs)

        # Set splitter proportions
        splitter.setSizes([500, 500])

        # Store state for two-step analysis
        self.detected_particles = None
        self.detection_worker = None
        self.tracking_worker = None

        # Connect actions
        self.connect_actions()

        # Add progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def open_file(self):
        """Open a movie file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Movie File",
            "",
            "Movie Files (*.tif *.tiff);;All Files (*.*)"
        )

        if file_path:
            try:
                # Load movie
                reader = DataReader()
                movie = reader.read_movie(file_path)

                if movie is None:
                    raise ValueError(f"Could not read movie file: {file_path}")

                # Set image data
                self.image_viewer.set_image_data(movie)

                # Store file path and enable actions
                self.current_file = file_path
                self.detect_action.setEnabled(True)  # Enable detect particles button
                self.run_action.setEnabled(True)

                # Clear previous results
                self.track_viewer.clear()
                self.feature_viewer.clear()

                # Set status
                self.status_bar.showMessage(f"Loaded: {file_path}")

                # Reset tracking button
                self.track_action.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def batch_process(self):
        """Open batch processing dialog"""
        # TODO: Implement batch processing dialog
        pass

    def run_analysis(self):
        """Run analysis on current file"""
        if not hasattr(self, 'current_file'):
            QMessageBox.warning(self, "Error", "No file loaded")
            return

        # Disable controls
        self.stop_action.setEnabled(True)

        # Get parameters
        parameters = {
            'min_sigma': self.min_sigma_spin.value(),
            'max_sigma': self.max_sigma_spin.value(),
            'threshold_rel': self.threshold_spin.value(),
            'max_distance': self.max_distance_spin.value(),
            'max_gap': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value()
        }

        # Create and start worker
        self.analysis_worker = AnalysisWorker(self.current_file, parameters)
        self.analysis_worker.progress.connect(self.update_progress)
        self.analysis_worker.finished.connect(self.analysis_finished)
        self.analysis_worker.error.connect(self.analysis_error)
        self.analysis_worker.start()

    def stop_analysis(self):
        """Stop current analysis"""
        if self.analysis_worker:
            self.analysis_worker.stop()
            self.stop_action.setEnabled(False)

    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)


    def analysis_finished(self, results: dict):
        """Handle analysis completion"""
        self.current_results = results
        self.stop_action.setEnabled(False)

        # Update particle overlay
        self.detected_particles = results['particles']
        self.image_viewer.set_particles(results['particles'])

        # Update viewers
        self.track_viewer.set_data(
            results['tracks'],
            results['features']
        )
        self.feature_viewer.set_features(results['features'])

        # Update results tables
        self.update_results_tables()

        # Update status
        n_particles = len(results['particles'])
        self.status_bar.showMessage(f"Analysis complete - Detected {n_particles} particles")

    def analysis_error(self, error_msg: str):
        """Handle analysis error"""
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")
        self.stop_action.setEnabled(False)
        self.status_bar.showMessage("Analysis failed")

    def closeEvent(self, event):
        """Handle application close"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Analysis is still running. Do you want to stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_analysis()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def run_detection(self):
        """Run particle detection on current ROI"""
        if not hasattr(self, 'current_file'):
            QMessageBox.warning(self, "Error", "No file loaded")
            return

        # Get ROI data
        roi_data, roi_bounds = self.image_viewer.get_roi_data()
        if roi_data is None:
            QMessageBox.warning(self, "Error", "Invalid ROI")
            return

        # Disable controls
        self.detect_action.setEnabled(False)
        self.stop_action.setEnabled(True)

        # Get detection parameters
        parameters = {
            'min_sigma': self.min_sigma_spin.value(),
            'max_sigma': self.max_sigma_spin.value(),
            'threshold_rel': self.threshold_spin.value()
        }

        # Create and start worker
        self.detection_worker = DetectionWorker(
            self.image_viewer.image_data,
            roi_bounds,
            parameters
        )
        self.detection_worker.progress.connect(self.update_progress)
        self.detection_worker.finished.connect(self.detection_finished)
        self.detection_worker.error.connect(self.analysis_error)
        self.detection_worker.start()

    def detection_finished(self, particles: list):
        """Handle completion of particle detection"""
        self.detected_particles = particles
        self.stop_action.setEnabled(False)
        self.detect_action.setEnabled(True)
        self.track_action.setEnabled(True)

        # Update display
        self.image_viewer.set_particles(particles)

        # Update status
        n_particles = len(particles)
        self.status_bar.showMessage(f"Detected {n_particles} particles")

    def run_tracking(self):
        """Run tracking and analysis on detected particles"""
        if self.detected_particles is None:
            QMessageBox.warning(self, "Error", "No particles detected")
            return

        # Disable controls
        self.track_action.setEnabled(False)
        self.stop_action.setEnabled(True)

        # Get tracking parameters
        parameters = {
            'max_distance': self.max_distance_spin.value(),
            'max_gap': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value()
        }

        # Create and start worker
        self.tracking_worker = TrackingWorker(
            self.detected_particles,
            parameters,
            Path(self.current_file).with_suffix('')
        )
        self.tracking_worker.progress.connect(self.update_progress)
        self.tracking_worker.finished.connect(self.tracking_finished)
        self.tracking_worker.error.connect(self.analysis_error)
        self.tracking_worker.start()

    def tracking_finished(self, results: dict):
        """Handle tracking completion"""
        self.current_results = results
        self.stop_action.setEnabled(False)
        self.track_action.setEnabled(True)

        # Update viewers
        self.track_viewer.set_data(
            results['tracks'],
            results['features']
        )
        self.feature_viewer.set_features(results['features'])

        # Update results tables
        self.update_results_tables()

        # Store particle data
        if 'particles' not in results:
            results['particles'] = self.detected_particles

        self.status_bar.showMessage(f"Tracking complete - {len(results['tracks'])} tracks identified")

    def stop_current(self):
        """Stop current operation"""
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
        self.stop_action.setEnabled(False)


    def setup_menubar(self):
        """Set up the application menu bar"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")

        open_action = file_menu.addAction("Open File...")
        open_action.triggered.connect(self.open_file)
        open_action.setShortcut("Ctrl+O")

        save_results_action = file_menu.addAction("Save Results...")
        save_results_action.triggered.connect(self.save_results)
        save_results_action.setShortcut("Ctrl+S")

        file_menu.addSeparator()

        export_movie_action = file_menu.addAction("Export Tracking Movie...")
        export_movie_action.triggered.connect(self.export_movie)

        file_menu.addSeparator()

        save_settings_action = file_menu.addAction("Save Settings...")
        save_settings_action.triggered.connect(self.save_settings)

        load_settings_action = file_menu.addAction("Load Settings...")
        load_settings_action.triggered.connect(self.load_settings)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")

        # Analysis Menu
        analysis_menu = menubar.addMenu("Analysis")

        detection_settings_action = analysis_menu.addAction("Particle Detection Settings...")
        detection_settings_action.triggered.connect(self.show_detection_settings)

        tracking_settings_action = analysis_menu.addAction("Tracking Settings...")
        tracking_settings_action.triggered.connect(self.show_tracking_settings)

        analysis_menu.addSeparator()

        batch_process_action = analysis_menu.addAction("Batch Process...")
        batch_process_action.triggered.connect(self.batch_process)

        # View Menu
        view_menu = menubar.addMenu("View")

        # Create overlay action
        self.show_overlay_action = view_menu.addAction("Show Particle Overlay")
        self.show_overlay_action.setCheckable(True)
        # Initialize checked state from the button
        self.show_overlay_action.setChecked(self.image_viewer.show_particles_btn.isChecked())
        # Connect the action to the button
        self.show_overlay_action.triggered.connect(self.image_viewer.show_particles_btn.setChecked)
        # And connect the button back to the action
        self.image_viewer.show_particles_btn.toggled.connect(self.show_overlay_action.setChecked)

        view_menu.addSeparator()

        self.show_tracks_action = view_menu.addAction("Show Tracks")
        self.show_tracks_action.setCheckable(True)
        self.show_tracks_action.setChecked(True)
        self.show_tracks_action.triggered.connect(self.track_viewer.setVisible)

        reset_view_action = view_menu.addAction("Reset View")
        reset_view_action.triggered.connect(self.reset_view)

        # Help Menu
        help_menu = menubar.addMenu("Help")

        help_action = help_menu.addAction("Help Contents")
        help_action.triggered.connect(self.show_help)
        help_action.setShortcut("F1")

        help_menu.addSeparator()

        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

        documentation_action = help_menu.addAction("Documentation")
        documentation_action.triggered.connect(self.show_documentation)

        # Optional: Add keyboard shortcuts
        shortcuts = QShortcut(QKeySequence("Ctrl+H"), self)
        shortcuts.activated.connect(self.show_help)

    def save_results(self):
        """Save analysis results"""
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "No Results", "No analysis results to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*.*)"
        )

        if file_path:
            try:
                writer = DataWriter()
                writer.write_analysis_summary(
                    self.current_results['tracks'],
                    self.current_results['features'],
                    file_path
                )
                self.status_bar.showMessage(f"Results saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")

    def export_movie(self):
        """Export tracking movie"""
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "No Results", "No tracks to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Movie",
            "",
            "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*.*)"
        )

        if file_path:
            try:
                from ..visualization.plot_utils import TrackVisualizer
                visualizer = TrackVisualizer()
                visualizer.create_track_movie(
                    self.current_results['tracks'],
                    file_path,
                    fps=10,
                    tail_length=10
                )
                self.status_bar.showMessage(f"Movie exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting movie: {str(e)}")

    def show_detection_settings(self):
        """Show particle detection settings dialog"""
        current_settings = {
            'min_sigma': self.min_sigma_spin.value(),
            'max_sigma': self.max_sigma_spin.value(),
            'threshold_rel': self.threshold_spin.value(),
            'exclude_border': True
        }

        dialog = SettingsDialog(current_settings, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            # Update spinboxes
            self.min_sigma_spin.setValue(new_settings['min_sigma'])
            self.max_sigma_spin.setValue(new_settings['max_sigma'])
            self.threshold_spin.setValue(new_settings['threshold_rel'])

    def show_tracking_settings(self):
        """Show particle tracking settings dialog"""
        current_settings = {
            'max_distance': self.max_distance_spin.value(),
            'max_gap': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value()
        }

        dialog = SettingsDialog(current_settings, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            # Update spinboxes
            self.max_distance_spin.setValue(new_settings['max_distance'])
            self.max_gap_spin.setValue(new_settings['max_gap'])
            self.min_length_spin.setValue(new_settings['min_track_length'])

    def toggle_overlay(self, checked: bool):
        """Toggle particle overlay visibility"""
        self.image_viewer.show_particles_btn.setChecked(checked)

    def reset_view(self):
        """Reset image view to default"""
        if self.image_viewer.image_data is not None:
            self.image_viewer.view_widget.setImage(self.image_viewer.image_data)
            self.image_viewer.update_display()


    def show_documentation(self):
        """Show documentation"""
        # Could open web browser to documentation or show PDF
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation available at: [Your documentation URL]"
        )

    def connect_actions(self):
        """Connect menu actions to slots"""
        # Update viewer connections
        self.image_viewer.frame_changed.connect(self.track_viewer.set_current_frame)
        self.track_viewer.track_selected.connect(self.show_track_details)

        # Note: Overlay connections will be made in setup_menubar

    def show_track_details(self, track_id: int):
        """Show details for selected track"""
        if hasattr(self, 'current_results') and self.current_results:
            track = next((t for t in self.current_results['tracks']
                         if t.id == track_id), None)
            if track:
                # Could show track details in a new dialog or panel
                self.status_bar.showMessage(f"Selected Track {track_id}: "
                                          f"{len(track.particles)} particles")

    def save_settings(self):
        """Save current settings to file"""
        settings = {
            'detection': {
                'min_sigma': self.min_sigma_spin.value(),
                'max_sigma': self.max_sigma_spin.value(),
                'threshold_rel': self.threshold_spin.value()
            },
            'tracking': {
                'max_distance': self.max_distance_spin.value(),
                'max_gap': self.max_gap_spin.value(),
                'min_track_length': self.min_length_spin.value()
            }
        }

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            try:
                import json
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=4)
                self.status_bar.showMessage(f"Settings saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving settings: {str(e)}")

    def load_settings(self):
        """Load settings from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Settings",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    settings = json.load(f)

                # Update detection settings
                det_settings = settings.get('detection', {})
                self.min_sigma_spin.setValue(det_settings.get('min_sigma', 1.0))
                self.max_sigma_spin.setValue(det_settings.get('max_sigma', 3.0))
                self.threshold_spin.setValue(det_settings.get('threshold_rel', 0.2))

                # Update tracking settings
                track_settings = settings.get('tracking', {})
                self.max_distance_spin.setValue(track_settings.get('max_distance', 5.0))
                self.max_gap_spin.setValue(track_settings.get('max_gap', 2))
                self.min_length_spin.setValue(track_settings.get('min_track_length', 3))

                self.status_bar.showMessage(f"Settings loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading settings: {str(e)}")

    def setup_results_tables(self):
        """Set up results table views"""
        # Create Tables tab
        tables_tab = QWidget()
        tables_layout = QVBoxLayout(tables_tab)

        # Create inner tab widget for tracks and features tables
        table_tabs = QTabWidget()

        # Tracks table
        self.tracks_table = QTableWidget()
        self.tracks_table.setColumnCount(7)
        self.tracks_table.setHorizontalHeaderLabels([
            'Track ID', 'Frame', 'X', 'Y', 'Intensity',
            'Start Frame', 'End Frame'
        ])
        table_tabs.addTab(self.tracks_table, "Tracks")

        # Features table
        self.features_table = QTableWidget()
        self.features_table.setColumnCount(10)
        self.features_table.setHorizontalHeaderLabels([
            'Track ID', 'Diffusion Coefficient', 'Alpha',
            'Radius Gyration', 'Asymmetry', 'Fractal Dimension',
            'Straightness', 'Mean Velocity', 'Mean Acceleration',
            'Confinement Ratio'
        ])
        table_tabs.addTab(self.features_table, "Features")

        tables_layout.addWidget(table_tabs)

        # Add filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Enter filter text...")
        self.filter_edit.textChanged.connect(self.filter_tables)
        filter_layout.addWidget(self.filter_edit)

        # Add export button
        export_btn = QPushButton("Export to Excel")
        export_btn.clicked.connect(self.export_tables)
        filter_layout.addWidget(export_btn)

        tables_layout.addLayout(filter_layout)

        return tables_tab

    def update_results_tables(self):
        """Update tables with current results"""
        if not hasattr(self, 'current_results') or not self.current_results:
            return

        # Update tracks table
        tracks = self.current_results['tracks']
        self.tracks_table.setRowCount(0)  # Clear existing rows
        row = 0
        for track in tracks:
            for particle in track.particles:
                self.tracks_table.insertRow(row)
                self.tracks_table.setItem(row, 0, QTableWidgetItem(str(track.id)))
                self.tracks_table.setItem(row, 1, QTableWidgetItem(str(particle.frame)))
                self.tracks_table.setItem(row, 2, QTableWidgetItem(f"{particle.x:.2f}"))
                self.tracks_table.setItem(row, 3, QTableWidgetItem(f"{particle.y:.2f}"))
                self.tracks_table.setItem(row, 4, QTableWidgetItem(f"{particle.intensity:.2f}"))
                self.tracks_table.setItem(row, 5, QTableWidgetItem(str(track.start_frame)))
                self.tracks_table.setItem(row, 6, QTableWidgetItem(str(track.end_frame)))
                row += 1

        # Update features table
        features = self.current_results['features']
        self.features_table.setRowCount(len(features))
        for i, feature in enumerate(features):
            self.features_table.setItem(i, 0, QTableWidgetItem(str(feature.track_id)))
            self.features_table.setItem(i, 1, QTableWidgetItem(f"{feature.diffusion_coefficient:.4f}"))
            self.features_table.setItem(i, 2, QTableWidgetItem(f"{feature.alpha:.4f}"))
            self.features_table.setItem(i, 3, QTableWidgetItem(f"{feature.radius_gyration:.4f}"))
            self.features_table.setItem(i, 4, QTableWidgetItem(f"{feature.asymmetry:.4f}"))
            self.features_table.setItem(i, 5, QTableWidgetItem(f"{feature.fractal_dimension:.4f}"))
            self.features_table.setItem(i, 6, QTableWidgetItem(f"{feature.straightness:.4f}"))
            self.features_table.setItem(i, 7, QTableWidgetItem(f"{feature.mean_velocity:.4f}"))
            self.features_table.setItem(i, 8, QTableWidgetItem(f"{feature.mean_acceleration:.4f}"))
            self.features_table.setItem(i, 9, QTableWidgetItem(f"{feature.confinement_ratio:.4f}"))

        # Enable sorting
        self.tracks_table.setSortingEnabled(True)
        self.features_table.setSortingEnabled(True)

    def filter_tables(self, text):
        """Filter table contents"""
        for table in [self.tracks_table, self.features_table]:
            for row in range(table.rowCount()):
                show_row = False
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    if item and text.lower() in item.text().lower():
                        show_row = True
                        break
                table.setRowHidden(row, not show_row)

    def export_tables(self):
        """Export tables to Excel"""
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tables",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*.*)"
        )

        if file_path:
            try:
                writer = DataWriter()
                writer.write_analysis_summary(
                    self.current_results['tracks'],
                    self.current_results['features'],
                    file_path
                )
                self.status_bar.showMessage(f"Tables exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting tables: {str(e)}")

    def show_help(self):
        """Show help dialog"""
        help_dialog = HelpDialog(self)
        help_dialog.exec()

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Particle Analysis",
            f"""<h3>Particle Analysis</h3>
            <p>Version: 1.0.0</p>
            <p>A tool for detecting and tracking particles in fluorescence microscopy data.</p>
            <p>Features:</p>
            <ul>
                <li>Particle detection using Gaussian fitting</li>
                <li>Particle tracking with nearest-neighbor linking</li>
                <li>Feature calculation and analysis</li>
                <li>Interactive visualization</li>
            </ul>
            <p>Created by: George Dickinson & Claude AI</p>
            """
        )
