# src/particle_analysis/gui/workers.py

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from ..core.particle_detection import ParticleDetector, Particle
from ..core.particle_tracking import ParticleTracker, Track
from ..core.feature_calculation import FeatureCalculator
from ..io.writers import DataWriter

# Set up logging
logger = logging.getLogger(__name__)

class DetectionWorker(QThread):
    """Worker thread for particle detection"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)  # List[Particle]
    error = pyqtSignal(str)

    def __init__(self,
                 image_data: np.ndarray,
                 roi_bounds: tuple,
                 parameters: dict):
        """
        Initialize detection worker

        Parameters
        ----------
        image_data : np.ndarray
            Image data to process
        roi_bounds : tuple
            (x_start, y_start, x_end, y_end) of ROI
        parameters : dict
            Detection parameters
        """
        super().__init__()
        self.image_data = image_data
        self.roi_bounds = roi_bounds
        self.parameters = parameters
        self.should_stop = False

    def run(self):
        try:
            # Extract ROI data
            x_start, y_start, x_end, y_end = self.roi_bounds
            roi_data = self.image_data[:, y_start:y_end, x_start:x_end]

            self.progress.emit(10)
            if self.should_stop:
                return

            # Create detector with parameters
            detector = ParticleDetector(
                min_sigma=self.parameters.get('min_sigma', 1.0),
                max_sigma=self.parameters.get('max_sigma', 3.0),
                threshold_rel=self.parameters.get('threshold_rel', 0.2)
            )

            # Detect particles
            particles = detector.detect_movie(roi_data)

            # Adjust particle coordinates for ROI
            for particle in particles:
                particle.x += x_start
                particle.y += y_start

            self.progress.emit(100)
            self.finished.emit(particles)

        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Detection error: {str(e)}")

    def stop(self):
        """Stop detection"""
        self.should_stop = True

class TrackingWorker(QThread):
    """Worker thread for particle tracking and analysis"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)  # Dict with tracks, features
    error = pyqtSignal(str)

    def __init__(self,
                 particles: list,
                 parameters: dict,
                 save_path: Optional[Path] = None):
        """
        Initialize tracking worker

        Parameters
        ----------
        particles : List[Particle]
            Detected particles to track
        parameters : dict
            Tracking parameters
        save_path : Path, optional
            Path to save results
        """
        super().__init__()
        self.particles = particles
        self.parameters = parameters
        self.save_path = save_path
        self.should_stop = False

    def run(self):
        try:
            # Track particles
            tracker = ParticleTracker(
                max_distance=self.parameters.get('max_distance', 5.0),
                max_gap=self.parameters.get('max_gap', 2),
                min_track_length=self.parameters.get('min_track_length', 3)
            )
            tracks = tracker.track_particles(self.particles)

            self.progress.emit(40)
            if self.should_stop:
                return

            # Calculate features
            calculator = FeatureCalculator()
            features = [calculator.calculate_track_features(track)
                       for track in tracks]
            features = [f for f in features if f is not None]

            self.progress.emit(80)
            if self.should_stop:
                return

            # Save results if path provided
            if self.save_path:
                writer = DataWriter()
                writer.write_tracks_csv(tracks, f"{self.save_path}_tracks.csv")
                writer.write_features_csv(features, f"{self.save_path}_features.csv")

            self.progress.emit(100)

            # Return results
            results = {
                'tracks': tracks,
                'features': features
            }
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Tracking error: {str(e)}")

    def stop(self):
        """Stop tracking"""
        self.should_stop = True
