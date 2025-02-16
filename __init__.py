"""
Initialization file for the particle_analysis package.
Provides high-level imports for core functionality.
"""

# Import key submodules for easier access
from .core import particle_detection, particle_tracking, feature_calculation
from .analysis import diffusion, statistics
from .io import readers, writers
from .visualization import plot_utils, viewers
from .gui import main_window, analysis_widget, viewers

# Define what gets imported with `from particle_analysis import *`
__all__ = [
    "particle_detection",
    "particle_tracking",
    "feature_calculation",
    "diffusion",
    "statistics",
    "readers",
    "writers",
    "plot_utils",
    "viewers",
    "main_window",
    "analysis_widget",
]
