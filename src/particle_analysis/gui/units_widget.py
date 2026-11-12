# src/particle_analysis/gui/units_widget.py
"""Explicit physical units for the motion analysis: pixel size (um/px) and frame interval (s).

Both start "not set" (no silent defaults): DiffusionAnalyzer works in um and seconds, so a motion analysis or export
done with a guessed scale would give numbers that look comparable and are not. The widget shows where the values
came from (file metadata, entered by hand) and builds the DiffusionAnalyzer only when both are set.
"""
from typing import Optional, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel

from ..analysis.diffusion import DiffusionAnalyzer

NOT_SET = "not set"


class UnitsWidget(QGroupBox):
    """Pixel size and frame interval inputs; changed() fires when either value changes."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Physical units (required for the motion analysis)", parent)
        layout = QHBoxLayout(self)
        self.pixel_spin = self._spin(4, " µm/px")
        self.interval_spin = self._spin(5, " s")
        for text, spin in (("Pixel size:", self.pixel_spin), ("Frame interval:", self.interval_spin)):
            label = QLabel(text)
            label.setBuddy(spin)
            layout.addWidget(label)
            layout.addWidget(spin)
        self.source_label = QLabel()
        layout.addWidget(self.source_label)
        layout.addStretch()
        self._source = None
        self._update_label()

    def _spin(self, decimals: int, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(0.0, 1e4)
        spin.setSingleStep(10 ** -(decimals - 1))
        spin.setSpecialValueText(NOT_SET)  # shown at the minimum (0): the value is unset
        spin.setSuffix(suffix)
        spin.setValue(0.0)
        spin.valueChanged.connect(self._edited)
        return spin

    def _edited(self):
        self._source = "entered"
        self._update_label()
        self.changed.emit()

    def set_units(self, pixel_size: Optional[float], frame_interval: Optional[float], source: Optional[str] = None):
        """Set the values (None leaves one unset), e.g. from the recording's metadata, and say where they came from."""
        for spin, value in ((self.pixel_spin, pixel_size), (self.interval_spin, frame_interval)):
            spin.blockSignals(True)
            spin.setValue(float(value) if value else 0.0)
            spin.blockSignals(False)
        self._source = source
        self._update_label()
        self.changed.emit()

    def units(self) -> Optional[Tuple[float, float]]:
        """(pixel size in um, frame interval in s), or None if either is not set."""
        px, dt = self.pixel_spin.value(), self.interval_spin.value()
        return (px, dt) if px > 0 and dt > 0 else None

    def analyzer(self) -> Optional[DiffusionAnalyzer]:
        """A DiffusionAnalyzer with these units, or None if they are not set."""
        u = self.units()
        return DiffusionAnalyzer(pixel_size=u[0], frame_interval=u[1]) if u else None

    def describe(self) -> str:
        u = self.units()
        if not u:
            return "units not set: set the pixel size and frame interval to run the motion analysis"
        return f"pixel size {u[0]:g} µm, frame interval {u[1]:g} s ({self._source or 'entered'})"

    def _update_label(self):
        self.source_label.setText(self.describe() if not self.units() else f"({self._source or 'entered'})")
