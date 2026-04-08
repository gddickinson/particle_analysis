"""
conftest.py - pytest configuration for particle_analysis tests.

Adds the src/ directory to sys.path so that 'particle_analysis' can be imported
without needing to install the package.
"""
import sys
from pathlib import Path

# Add src/ to path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
