# src/particle_analysis/gui/main_window.py

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QFileDialog, QTabWidget, QLabel,
                            QProgressBar, QMessageBox, QSpinBox, QComboBox,
                            QCheckBox, QGroupBox, QToolBar, QStatusBar, QDoubleSpinBox, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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
        super().__init__()
        self.setWindowTitle("Particle Analysis")
        self.setup_ui()

        # Initialize state
        self.current_results = None
        self.analysis_worker = None

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

        run_action = toolbar.addAction("Run Analysis")
        run_action.triggered.connect(self.run_analysis)
        run_action.setEnabled(False)
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
        tabs.addTab(self.track_viewer, "Tracks")

        # Feature analysis tab
        self.feature_viewer = FeatureViewer()
        tabs.addTab(self.feature_viewer, "Features")

        splitter.addWidget(tabs)

        # Set splitter proportions
        splitter.setSizes([500, 500])

        # Store state for two-step analysis
        self.detected_particles = None
        self.detection_worker = None
        self.tracking_worker = None

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

        # Update viewers
        self.track_viewer.set_data(
            results['tracks'],
            results['features']
        )
        self.feature_viewer.set_features(results['features'])

        self.status_bar.showMessage("Analysis complete")

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

        self.status_bar.showMessage("Analysis complete")

    def stop_current(self):
        """Stop current operation"""
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
        self.stop_action.setEnabled(False)
