# src/particle_analysis/gui/settings_dialog.py

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QSpinBox, QDoubleSpinBox, QTabWidget,
                            QWidget, QGroupBox, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt
import json
from pathlib import Path
from typing import Dict
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SettingsDialog(QDialog):
    """Dialog for configuring analysis parameters"""

    def __init__(self, current_settings: Dict = None, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Analysis Settings")
        layout = QVBoxLayout(self)

        # Create tabs for different settings categories
        tabs = QTabWidget()

        # Detection settings tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)

        # Particle detection group
        det_group = QGroupBox("Particle Detection")
        det_layout = QVBoxLayout()

        # Sigma range
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma Range:"))

        self.min_sigma_spin = QDoubleSpinBox()
        self.min_sigma_spin.setRange(0.5, 10.0)
        self.min_sigma_spin.setSingleStep(0.1)
        self.min_sigma_spin.setValue(1.0)
        sigma_layout.addWidget(self.min_sigma_spin)

        sigma_layout.addWidget(QLabel("to"))

        self.max_sigma_spin = QDoubleSpinBox()
        self.max_sigma_spin.setRange(1.0, 20.0)
        self.max_sigma_spin.setSingleStep(0.1)
        self.max_sigma_spin.setValue(3.0)
        sigma_layout.addWidget(self.max_sigma_spin)

        det_layout.addLayout(sigma_layout)

        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Detection Threshold:"))

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.2)
        threshold_layout.addWidget(self.threshold_spin)

        det_layout.addLayout(threshold_layout)

        # Additional detection options
        self.exclude_border_check = QCheckBox("Exclude Border Particles")
        self.exclude_border_check.setChecked(True)
        det_layout.addWidget(self.exclude_border_check)

        det_group.setLayout(det_layout)
        detection_layout.addWidget(det_group)

        tabs.addTab(detection_tab, "Detection")

        # Tracking settings tab
        tracking_tab = QWidget()
        tracking_layout = QVBoxLayout(tracking_tab)

        # Particle tracking group
        track_group = QGroupBox("Particle Tracking")
        track_layout = QVBoxLayout()

        # Max distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Maximum Distance:"))

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(1.0, 50.0)
        self.max_distance_spin.setSingleStep(0.5)
        self.max_distance_spin.setValue(5.0)
        dist_layout.addWidget(self.max_distance_spin)

        track_layout.addLayout(dist_layout)

        # Max gap
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Maximum Gap:"))

        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 10)
        self.max_gap_spin.setValue(2)
        gap_layout.addWidget(self.max_gap_spin)

        track_layout.addLayout(gap_layout)

        # Min track length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Minimum Track Length:"))

        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(2, 20)
        self.min_length_spin.setValue(3)
        length_layout.addWidget(self.min_length_spin)

        track_layout.addLayout(length_layout)

        # Link method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Linking Method:"))

        self.link_method_combo = QComboBox()
        self.link_method_combo.addItems([
            "Nearest Neighbor",
            "Hungarian Algorithm"
        ])
        method_layout.addWidget(self.link_method_combo)

        track_layout.addLayout(method_layout)

        track_group.setLayout(track_layout)
        tracking_layout.addWidget(track_group)

        tabs.addTab(tracking_tab, "Tracking")

        # Analysis settings tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)

        # Feature calculation group
        feature_group = QGroupBox("Feature Calculation")
        feature_layout = QVBoxLayout()

        # MSD settings
        msd_layout = QHBoxLayout()
        msd_layout.addWidget(QLabel("Max MSD Points:"))

        self.max_msd_spin = QSpinBox()
        self.max_msd_spin.setRange(5, 50)
        self.max_msd_spin.setValue(10)
        msd_layout.addWidget(self.max_msd_spin)

        feature_layout.addLayout(msd_layout)

        # Feature selection
        self.feature_checks = {}
        features = [
            "MSD Analysis",
            "Radius of Gyration",
            "Asymmetry",
            "Fractal Dimension",
            "Velocity Analysis"
        ]

        for feature in features:
            check = QCheckBox(feature)
            check.setChecked(True)
            self.feature_checks[feature] = check
            feature_layout.addWidget(check)

        feature_group.setLayout(feature_layout)
        analysis_layout.addWidget(feature_group)

        tabs.addTab(analysis_tab, "Analysis")

        # Add tabs to layout
        layout.addWidget(tabs)

        # Add buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def load_settings(self):
        """Load current settings into UI"""
        # Detection settings
        self.min_sigma_spin.setValue(
            self.current_settings.get('min_sigma', 1.0))
        self.max_sigma_spin.setValue(
            self.current_settings.get('max_sigma', 3.0))
        self.threshold_spin.setValue(
            self.current_settings.get('threshold_rel', 0.2))
        self.exclude_border_check.setChecked(
            self.current_settings.get('exclude_border', True))

        # Tracking settings
        self.max_distance_spin.setValue(
            self.current_settings.get('max_distance', 5.0))
        self.max_gap_spin.setValue(
            self.current_settings.get('max_gap', 2))
        self.min_length_spin.setValue(
            self.current_settings.get('min_track_length', 3))

        method_idx = self.link_method_combo.findText(
            self.current_settings.get('link_method', "Nearest Neighbor"))
        if method_idx >= 0:
            self.link_method_combo.setCurrentIndex(method_idx)

        # Analysis settings
        self.max_msd_spin.setValue(
            self.current_settings.get('max_msd_points', 10))

        features = self.current_settings.get('features', {})
        for feature, check in self.feature_checks.items():
            check.setChecked(features.get(feature, True))

    def get_settings(self) -> Dict:
        """Get current settings from UI"""
        settings = {
            # Detection settings
            'min_sigma': self.min_sigma_spin.value(),
            'max_sigma': self.max_sigma_spin.value(),
            'threshold_rel': self.threshold_spin.value(),
            'exclude_border': self.exclude_border_check.isChecked(),

            # Tracking settings
            'max_distance': self.max_distance_spin.value(),
            'max_gap': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value(),
            'link_method': self.link_method_combo.currentText(),

            # Analysis settings
            'max_msd_points': self.max_msd_spin.value(),
            'features': {
                feature: check.isChecked()
                for feature, check in self.feature_checks.items()
            }
        }

        return settings

    def reset_settings(self):
        """Reset settings to defaults"""
        self.current_settings = {}
        self.load_settings()

    def save_settings(self, file_path: str):
        """Save settings to file"""
        try:
            settings = self.get_settings()
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
            return False

    @staticmethod
    def load_settings_file(file_path: str) -> Dict:
        """Load settings from file"""
        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
            return settings
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            return {}
