# API Reference

## Core Module

### ParticleDetector

```python
class ParticleDetector:
    """Class for detecting particles in fluorescence microscopy images"""
    
    def __init__(self, min_sigma=1.0, max_sigma=3.0, sigma_steps=5,
                 threshold_rel=0.2, min_distance=5, exclude_border=True):
        """Initialize detector with parameters"""
        
    def detect_frame(self, frame: np.ndarray) -> List[Particle]:
        """Detect particles in a single frame"""
        
    def detect_movie(self, movie: np.ndarray) -> List[Particle]:
        """Detect particles in all frames of a movie"""
```

### ParticleTracker

```python
class ParticleTracker:
    """Class for tracking particles across frames"""
    
    def __init__(self, max_distance=5.0, max_gap=2, min_track_length=3):
        """Initialize tracker with parameters"""
        
    def track_particles(self, particles: List[Particle]) -> List[Track]:
        """Link particles across frames to form tracks"""
```

### FeatureCalculator

```python
class FeatureCalculator:
    """Class for calculating track features"""
    
    def __init__(self, max_msd_points=10, min_track_length=5):
        """Initialize calculator with parameters"""
        
    def calculate_track_features(self, track: Track) -> Optional[TrackFeatures]:
        """Calculate features for a single track"""
```

## Data Structures

### Particle

```python
@dataclass
class Particle:
    """Data class for storing particle information"""
    frame: int
    x: float
    y: float
    intensity: float
    sigma: float
    snr: float
    frame_size: Tuple[int, int]
    id: Optional[int] = None
```

### Track

```python
@dataclass
class Track:
    """Data class for storing particle track information"""
    id: int
    particles: List[Particle]
    start_frame: int
    end_frame: int
```

### TrackFeatures

```python
@dataclass
class TrackFeatures:
    """Data class for storing calculated track features"""
    track_id: int
    msd_values: np.ndarray
    diffusion_coefficient: float
    alpha: float
    radius_gyration: float
    asymmetry: float
    fractal_dimension: float
    straightness: float
    mean_velocity: float
    mean_acceleration: float
    confinement_ratio: float
```

## I/O Module

### DataReader

```python
class DataReader:
    """Class for reading various data formats"""
    
    def __init__(self, pixel_size=0.108):
        """Initialize reader with microscope parameters"""
        
    def read_movie(self, file_path: Union[str, Path],
                   channel: int = 0) -> Optional[np.ndarray]:
        """Read movie file (TIFF stack)"""
```

### DataWriter

```python
class DataWriter:
    """Class for writing various data formats"""
    
    def write_tracks_csv(self, tracks: List[Track],
                        file_path: Union[str, Path]) -> bool:
        """Write track data to CSV"""
        
    def write_features_csv(self, features: List[TrackFeatures],
                          file_path: Union[str, Path]) -> bool:
        """Write feature data to CSV"""
```

## Analysis Module

### DiffusionAnalyzer

```python
class DiffusionAnalyzer:
    """Class for analyzing particle diffusion"""
    
    def calculate_msd(self, track: Track,
                     max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Mean Square Displacement curve"""
        
    def classify_motion(self, track: Track) -> MotionType:
        """Classify type of motion based on MSD analysis"""
```

## GUI Module

See the User Guide for details on using the graphical interface components.