# Development Guide

## Environment Setup

### Development Environment

1. Install development dependencies:
```bash
pip install -r requirements/requirements-dev.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### IDE Configuration

#### VSCode
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true
}
```

#### PyCharm
- Enable PEP 8 checking
- Set Black as formatter
- Enable type checking

## Testing

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_particle_detection.py
```

With coverage:
```bash
pytest --cov=src tests/
```

### Writing Tests

Example test structure:
```python
import pytest
from particle_analysis.core.particle_detection import ParticleDetector

def test_particle_detection():
    # Arrange
    detector = ParticleDetector()
    test_image = create_test_image()
    
    # Act
    particles = detector.detect_frame(test_image)
    
    # Assert
    assert len(particles) > 0
    assert all(p.intensity > 0 for p in particles)
```

## Performance Profiling

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

### Time Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
```

## GUI Development

### Adding New Features

1. Create new widget class:
```python
class NewFeatureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # UI setup code
        pass
```

2. Add to main window:
```python
self.new_feature = NewFeatureWidget()
layout.addWidget(self.new_feature)
```

### Qt Style Guidelines

- Use Qt Designer for complex layouts
- Follow Qt naming conventions
- Connect signals in `setup_ui`
- Use layouts instead of fixed positions

## Debugging Tips

### Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Debugging GUI

- Use QDebug for Qt-specific issues
- Enable Qt debug output
- Use Qt Creator for visual debugging

### Common Issues

1. Memory Leaks
- Use weak references
- Properly delete Qt objects
- Monitor memory usage

2. Performance
- Profile critical sections
- Use Qt's built-in profiling tools
- Monitor CPU and memory usage

3. Thread Safety
- Use Qt's thread mechanisms
- Avoid direct widget access from threads
- Use signals and slots for thread communication

## Documentation

### Building Documentation

```bash
cd docs
make html
```

### Documentation Style

Example docstring:
```python
def analyze_track(track: Track, parameters: Dict) -> TrackFeatures:
    """
    Analyze a particle track and calculate features.
    
    Parameters
    ----------
    track : Track
        Track object containing particle positions
    parameters : Dict
        Analysis parameters
        
    Returns
    -------
    TrackFeatures
        Calculated track features
        
    Examples
    --------
    >>> analyzer = TrackAnalyzer()
    >>> features = analyzer.analyze_track(track, {'max_points': 10})
    """
    # Implementation
```

## Release Process

1. Update version:
```bash
bump2version patch  # or minor, major
```

2. Generate changelog:
```bash
gitchangelog > CHANGELOG.md
```

3. Build distribution:
```bash
python -m build
```

4. Upload to PyPI:
```bash
twine upload dist/*
```