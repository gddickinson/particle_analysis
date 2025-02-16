# src/particle_analysis/gui/image_viewer.py

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import numpy as np

class ImageViewer(QWidget):
    """Widget for displaying and navigating image stacks"""

    # Signals
    frame_changed = pyqtSignal(int)  # Emitted when current frame changes
    region_selected = pyqtSignal(tuple)  # Emitted when ROI is selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data = None
        self.current_frame = 0
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)

        # Create image display
        self.view_widget = pg.ImageView()
        self.view_widget.ui.roiBtn.hide()
        self.view_widget.ui.menuBtn.hide()
        layout.addWidget(self.view_widget)

        # Create particle overlay handler
        self.particle_overlay = ParticleOverlay(self.view_widget)

        # Create particle control panel
        control_panel = QHBoxLayout()

        # Particle display toggle
        self.show_particles_btn = QPushButton("Show Particles")
        self.show_particles_btn.setCheckable(True)
        self.show_particles_btn.setEnabled(False)
        self.show_particles_btn.toggled.connect(self.particle_overlay.set_visible)
        control_panel.addWidget(self.show_particles_btn)

        # Frame navigation
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.valueChanged.connect(self.set_current_frame)
        frame_layout.addWidget(self.frame_spinbox)

        # Display controls
        frame_layout.addWidget(QLabel("Levels:"))
        self.min_level_spin = QSpinBox()
        self.max_level_spin = QSpinBox()
        frame_layout.addWidget(self.min_level_spin)
        frame_layout.addWidget(self.max_level_spin)

        control_panel.addLayout(frame_layout)
        layout.addLayout(control_panel)

        # Add ROI
        self.roi = pg.ROI([0, 0], [100, 100], pen=(0,9))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.view_widget.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.roi_changed)

    def set_particles(self, particles):
        """Set particles for display"""
        self.particle_overlay.set_particles(particles)
        self.show_particles_btn.setEnabled(len(particles) > 0)
        if len(particles) > 0:
            self.show_particles_btn.setChecked(True)

    def set_image_data(self, data: np.ndarray):
        """Set image data to display"""
        self.image_data = data

        if data is not None:
            # Update frame range
            self.frame_spinbox.setMaximum(data.shape[0] - 1)

            # Update level controls
            data_min = int(np.min(data))
            data_max = int(np.max(data))
            self.min_level_spin.setRange(data_min, data_max)
            self.max_level_spin.setRange(data_min, data_max)
            self.min_level_spin.setValue(data_min)
            self.max_level_spin.setValue(data_max)

            # Update display
            self.view_widget.setImage(data, axes={'t': 0, 'y': 1, 'x': 2})

            # Set ROI to center of image
            center_y = data.shape[1] // 2
            center_x = data.shape[2] // 2
            roi_size = min(data.shape[1], data.shape[2]) // 4
            self.roi.setPos([center_x - roi_size//2, center_y - roi_size//2])
            self.roi.setSize([roi_size, roi_size])

    def set_current_frame(self, frame):
        """Set current frame"""
        if self.image_data is not None and 0 <= frame < self.image_data.shape[0]:
            self.current_frame = frame
            self.view_widget.setCurrentIndex(frame)
            self.particle_overlay.set_frame(frame)
            self.frame_changed.emit(frame)

    def update_levels(self):
        """Update image display levels"""
        if self.image_data is not None:
            self.view_widget.setLevels(
                min=self.min_level_spin.value(),
                max=self.max_level_spin.value()
            )

    def roi_changed(self):
        """Handle ROI region change"""
        if self.image_data is not None:
            pos = self.roi.pos()
            size = self.roi.size()
            self.region_selected.emit((pos, size))

    def get_roi_data(self) -> tuple:
        """Get data within ROI for current frame"""
        if self.image_data is None:
            return None

        # Get ROI bounds
        pos = self.roi.pos()
        size = self.roi.size()
        x_start = int(max(0, pos[0]))
        y_start = int(max(0, pos[1]))
        x_end = int(min(self.image_data.shape[2], x_start + size[0]))
        y_end = int(min(self.image_data.shape[1], y_start + size[1]))

        # Extract ROI data
        roi_data = self.image_data[self.current_frame,
                                  y_start:y_end,
                                  x_start:x_end]

        return roi_data, (x_start, y_start, x_end, y_end)

    def clear(self):
        """Clear display"""
        self.image_data = None
        self.particle_overlay.clear()
        self.show_particles_btn.setEnabled(False)
        self.show_particles_btn.setChecked(False)
        self.view_widget.clear()

class ParticleOverlay:
    """Class for handling particle overlay visualization"""

    # Constants for particle display
    PARTICLE_SIZE = 20
    PARTICLE_COLOR = (255, 0, 0)  # Red

    def __init__(self, view_widget):
        self.view_widget = view_widget
        self.particles = []
        self.scatter_item = None
        self.current_frame = 0
        self.visible = False

    def set_particles(self, particles):
        """Set particles to display"""
        self.particles = particles
        self.update_display()

    def set_frame(self, frame):
        """Update display for given frame"""
        self.current_frame = frame
        self.update_display()

    def set_visible(self, visible):
        """Toggle visibility"""
        self.visible = visible
        self.update_display()

    def update_display(self):
        """Update particle overlay"""
        # Remove existing scatter plot if any
        if self.scatter_item is not None:
            self.view_widget.removeItem(self.scatter_item)
            self.scatter_item = None

        if self.visible and self.particles:
            # Get particles in current frame
            frame_particles = [p for p in self.particles if p.frame == self.current_frame]

            if frame_particles:
                # Create positions array and scale sizes by intensity
                positions = np.array([[p.x, p.y] for p in frame_particles])
                intensities = np.array([p.intensity for p in frame_particles])
                sizes = intensities / np.max(intensities) * self.PARTICLE_SIZE

                # Create scatter plot
                self.scatter_item = pg.ScatterPlotItem(
                    pos=positions,
                    size=sizes,
                    pen=pg.mkPen(self.PARTICLE_COLOR),
                    brush=pg.mkBrush(None),
                    symbol='o'
                )
                self.view_widget.addItem(self.scatter_item)

    def clear(self):
        """Clear all particles"""
        self.particles = []
        if self.scatter_item is not None:
            self.view_widget.removeItem(self.scatter_item)
            self.scatter_item = None
