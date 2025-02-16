# src/particle_analysis/gui/batch_dialog.py

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                            QListWidget, QFileDialog, QLabel, QProgressBar,
                            QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from typing import List, Dict
import logging

from ..core.particle_detection import ParticleDetector
from ..core.particle_tracking import ParticleTracker
from ..core.feature_calculation import FeatureCalculator
from ..io.readers import DataReader
from ..io.writers import DataWriter

# Set up logging
logger = logging.getLogger(__name__)

class BatchProcessor(QThread):
    """Worker thread for batch processing"""
    progress = pyqtSignal(int, int)  # file_index, progress
    file_finished = pyqtSignal(int, str)  # file_index, status
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(int, str)  # file_index, error message

    def __init__(self,
                 file_paths: List[str],
                 parameters: dict,
                 output_dir: str):
        super().__init__()
        self.file_paths = file_paths
        self.parameters = parameters
        self.output_dir = output_dir
        self.should_stop = False

    def run(self):
        results = {}

        for i, file_path in enumerate(self.file_paths):
            if self.should_stop:
                break

            try:
                # Process single file
                result = self.process_file(file_path, i)
                if result:
                    results[file_path] = result
                    self.file_finished.emit(i, "Success")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                self.error.emit(i, str(e))
                results[file_path] = {
                    'status': 'error',
                    'error_message': str(e)
                }

        # Save batch results
        if results:
            writer = DataWriter()
            writer.write_batch_results(
                results,
                Path(self.output_dir) / "batch_results.xlsx"
            )

        self.finished.emit(results)


    def process_file(self, file_path: str, file_index: int) -> Dict:
        """Process a single file"""
        # Load data
        reader = DataReader()
        movie = reader.read_movie(file_path)
        if movie is None:
            raise ValueError(f"Could not read movie file: {file_path}")

        self.progress.emit(file_index, 20)

        # Detect particles
        detector = ParticleDetector(
            min_sigma=self.parameters.get('min_sigma', 1.0),
            max_sigma=self.parameters.get('max_sigma', 3.0),
            threshold_rel=self.parameters.get('threshold_rel', 0.2)
        )
        particles = detector.detect_movie(movie)

        self.progress.emit(file_index, 40)

        # Track particles
        tracker = ParticleTracker(
            max_distance=self.parameters.get('max_distance', 5.0),
            max_gap=self.parameters.get('max_gap', 2),
            min_track_length=self.parameters.get('min_track_length', 3)
        )
        tracks = tracker.track_particles(particles)

        self.progress.emit(file_index, 60)

        # Calculate features
        calculator = FeatureCalculator()
        features = [calculator.calculate_track_features(track)
                   for track in tracks]
        features = [f for f in features if f is not None]

        self.progress.emit(file_index, 80)

        # Save results
        writer = DataWriter()
        output_path = Path(self.output_dir) / Path(file_path).name
        base_path = output_path.with_suffix('')

        writer.write_tracks_csv(tracks, f"{base_path}_tracks.csv")
        writer.write_features_csv(features, f"{base_path}_features.csv")

        self.progress.emit(file_index, 100)

        # Return results summary
        return {
            'status': 'success',
            'n_particles': len(particles),
            'n_tracks': len(tracks),
            'n_features': len(features),
            'mean_track_length': np.mean([len(t.particles) for t in tracks])
            if tracks else 0,
            'processing_time': None  # TODO: Add timing
        }

    def stop(self):
        """Stop batch processing"""
        self.should_stop = True

class BatchDialog(QDialog):
    """Dialog for batch processing multiple files"""

    def __init__(self, parameters: dict, parent=None):
        super().__init__(parent)
        self.parameters = parameters
        self.file_paths = []
        self.processor = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Batch Processing")
        layout = QVBoxLayout(self)

        # File list
        file_group = QHBoxLayout()
        self.file_list = QListWidget()
        file_group.addWidget(self.file_list)

        file_buttons = QVBoxLayout()
        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self.add_files)
        file_buttons.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_files)
        file_buttons.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_files)
        file_buttons.addWidget(clear_btn)

        file_group.addLayout(file_buttons)
        layout.addLayout(file_group)

        # Output directory selection
        output_group = QHBoxLayout()
        output_group.addWidget(QLabel("Output Directory:"))
        self.output_label = QLabel("Not selected")
        output_group.addWidget(self.output_label)

        output_btn = QPushButton("Select")
        output_btn.clicked.connect(self.select_output)
        output_group.addWidget(output_btn)

        layout.addLayout(output_group)

        # Progress indicators
        self.progress_bars = {}
        self.progress_widget = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_widget)
        layout.addWidget(self.progress_widget)

        # Control buttons
        button_group = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        button_group.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        button_group.addWidget(self.stop_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_group.addWidget(self.close_btn)

        layout.addLayout(button_group)

    def add_files(self):
        """Add files for processing"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Movie Files",
            "",
            "Movie Files (*.tif *.tiff);;All Files (*.*)"
        )

        if files:
            self.file_paths.extend(files)
            self.file_list.clear()
            self.file_list.addItems([Path(f).name for f in self.file_paths])

            # Add progress bars
            for file_path in files:
                if file_path not in self.progress_bars:
                    progress_layout = QHBoxLayout()
                    progress_layout.addWidget(QLabel(Path(file_path).name))

                    progress_bar = QProgressBar()
                    progress_bar.setValue(0)
                    progress_layout.addWidget(progress_bar)

                    status_label = QLabel("Pending")
                    progress_layout.addWidget(status_label)

                    self.progress_bars[file_path] = {
                        'bar': progress_bar,
                        'label': status_label
                    }
                    self.progress_layout.addLayout(progress_layout)

            self.update_start_button()

    def remove_files(self):
        """Remove selected files"""
        for item in self.file_list.selectedItems():
            file_path = self.file_paths[self.file_list.row(item)]
            self.file_paths.remove(file_path)

            # Remove progress bar
            if file_path in self.progress_bars:
                progress_bar = self.progress_bars[file_path]['bar']
                status_label = self.progress_bars[file_path]['label']
                self.progress_layout.removeWidget(progress_bar)
                self.progress_layout.removeWidget(status_label)
                progress_bar.deleteLater()
                status_label.deleteLater()
                del self.progress_bars[file_path]

        self.file_list.clear()
        self.file_list.addItems([Path(f).name for f in self.file_paths])
        self.update_start_button()

    def clear_files(self):
        """Clear all files"""
        self.file_paths = []
        self.file_list.clear()

        # Clear progress bars
        for widgets in self.progress_bars.values():
            widgets['bar'].deleteLater()
            widgets['label'].deleteLater()
        self.progress_bars = {}

        self.update_start_button()

    def select_output(self):
        """Select output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )

        if directory:
            self.output_label.setText(directory)
            self.update_start_button()

    def update_start_button(self):
        """Update start button enabled state"""
        self.start_btn.setEnabled(
            len(self.file_paths) > 0 and
            self.output_label.text() != "Not selected"
        )

    def start_processing(self):
        """Start batch processing"""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Create and start processor
        self.processor = BatchProcessor(
            self.file_paths,
            self.parameters,
            self.output_label.text()
        )
        self.processor.progress.connect(self.update_progress)
        self.processor.file_finished.connect(self.file_finished)
        self.processor.error.connect(self.file_error)
        self.processor.finished.connect(self.processing_finished)
        self.processor.start()

    def stop_processing(self):
        """Stop batch processing"""
        if self.processor:
            self.processor.stop()
            self.stop_btn.setEnabled(False)

    def update_progress(self, file_index: int, progress: int):
        """Update progress for a file"""
        file_path = self.file_paths[file_index]
        if file_path in self.progress_bars:
            self.progress_bars[file_path]['bar'].setValue(progress)

    def file_finished(self, file_index: int, status: str):
        """Handle file processing completion"""
        file_path = self.file_paths[file_index]
        if file_path in self.progress_bars:
            self.progress_bars[file_path]['label'].setText(status)

    def file_error(self, file_index: int, error_msg: str):
        """Handle file processing error"""
        file_path = self.file_paths[file_index]
        if file_path in self.progress_bars:
            self.progress_bars[file_path]['label'].setText("Error")

        QMessageBox.warning(
            self,
            "Processing Error",
            f"Error processing {Path(file_path).name}: {error_msg}"
        )

    def processing_finished(self, results: dict):
        """Handle batch processing completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processed {len(results)} files.\nResults saved to {self.output_label.text()}"
        )

    def closeEvent(self, event):
        """Handle dialog close"""
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is still running. Do you want to stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
