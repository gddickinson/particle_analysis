# tests/test_feature_calculation.py

import pytest
import numpy as np
from particle_analysis.core.feature_calculation import FeatureCalculator, TrackFeatures
from particle_analysis.core.particle_tracking import Track
from particle_analysis.core.particle_detection import Particle

@pytest.fixture
def calculator():
    """Create feature calculator with default parameters"""
    return FeatureCalculator()

@pytest.fixture
def linear_track():
    """Create a track with linear motion"""
    particles = []
    for i in range(10):
        particles.append(
            Particle(
                frame=i,
                x=float(i),  # Moving right at constant speed
                y=0.0,       # Constant y
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    return Track(id=1, particles=particles, start_frame=0, end_frame=9)

@pytest.fixture
def circular_track():
    """Create a track with circular motion"""
    particles = []
    for i in range(20):
        angle = 2 * np.pi * i / 19  # Full circle
        particles.append(
            Particle(
                frame=i,
                x=10 * np.cos(angle),  # Radius 10 circle
                y=10 * np.sin(angle),
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    return Track(id=2, particles=particles, start_frame=0, end_frame=19)

@pytest.fixture
def random_track():
    """Create a track with random motion"""
    np.random.seed(42)  # For reproducibility
    particles = []
    x, y = 0, 0
    for i in range(15):
        x += np.random.normal(0, 1)
        y += np.random.normal(0, 1)
        particles.append(
            Particle(
                frame=i,
                x=x,
                y=y,
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    return Track(id=3, particles=particles, start_frame=0, end_frame=14)

def test_feature_dataclass():
    """Test TrackFeatures dataclass"""
    features = TrackFeatures(
        track_id=1,
        msd_values=np.array([1.0, 2.0, 3.0]),
        diffusion_coefficient=0.5,
        alpha=1.0,
        radius_gyration=2.0,
        asymmetry=0.1,
        fractal_dimension=1.5,
        straightness=0.9,
        mean_velocity=1.0,
        mean_acceleration=0.1,
        confinement_ratio=0.8
    )
    
    # Test basic properties
    assert features.track_id == 1
    assert len(features.msd_values) == 3
    assert features.diffusion_coefficient == 0.5
    
    # Test dictionary conversion
    feature_dict = features.to_dict()
    assert isinstance(feature_dict, dict)
    assert feature_dict['track_id'] == 1
    assert len(feature_dict['msd_values']) == 3

def test_linear_motion(calculator, linear_track):
    """Test feature calculation for linear motion"""
    features = calculator.calculate_track_features(linear_track)
    
    assert features is not None
    # Linear motion should have straightness close to 1
    assert features.straightness > 0.95
    # asymmetry = 1 - lambda_min/lambda_max of the position covariance: 1 for a straight line
    # (the old expectation "< 0.1" contradicted the definition; the circle test checks the low end)
    assert features.asymmetry > 0.9
    # Alpha should be close to 2 for ballistic motion
    assert np.abs(features.alpha - 2.0) < 0.2

def test_circular_motion(calculator, circular_track):
    """Test feature calculation for circular motion"""
    features = calculator.calculate_track_features(circular_track)
    
    assert features is not None
    # Circular motion should have low straightness
    assert features.straightness < 0.1
    # Circular motion should have low asymmetry
    assert features.asymmetry < 0.1
    # Radius of gyration should be close to circle radius
    assert np.abs(features.radius_gyration - 10.0) < 1.0

def test_random_motion(calculator, random_track):
    """Test feature calculation for random motion"""
    features = calculator.calculate_track_features(random_track)
    
    assert features is not None
    # Random motion should have alpha close to 1 (diffusive)
    assert np.abs(features.alpha - 1.0) < 0.3
    # Random motion should have intermediate straightness
    assert 0.2 < features.straightness < 0.8


@pytest.mark.xfail(strict=True, reason=(
    "Box counting over 15 points between min/2 and 2*max pairwise distance gives ~0.5, not 2. "
    "Whether to change the estimator is a method decision for George (see SESSION_LOG.md)."))
def test_random_walk_fractal_dimension(calculator, random_track):
    """Fractal dimension should be close to 2 for a random walk (original expectation, kept as xfail)."""
    features = calculator.calculate_track_features(random_track)
    assert np.abs(features.fractal_dimension - 2.0) < 0.3

def test_msd_calculation(calculator):
    """Test MSD calculation specifically"""
    # Create a track with known displacements
    particles = []
    for i in range(5):
        particles.append(
            Particle(
                frame=i,
                x=float(i),  # Unit displacement each step
                y=0.0,
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    track = Track(id=1, particles=particles, start_frame=0, end_frame=4)
    
    positions = track.positions
    times = track.frames
    
    msd_values, time_lags = calculator.calculate_msd(positions, times)
    
    # For unit displacements, MSD should increase quadratically
    expected_msd = np.array([1., 4., 9., 16.])
    assert np.allclose(msd_values, expected_msd, rtol=0.1)

def test_minimum_track_length(calculator):
    """Test minimum track length requirement"""
    # Create a short track
    particles = [
        Particle(frame=i, x=float(i), y=0.0, intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
        for i in range(3)  # Less than minimum length
    ]
    track = Track(id=1, particles=particles, start_frame=0, end_frame=2)
    
    # Should return None for too short tracks
    features = calculator.calculate_track_features(track)
    assert features is None

def test_error_handling(calculator):
    """Test error handling in feature calculation"""
    # Test with invalid positions
    particles = [
        Particle(frame=i, x=np.nan, y=0.0, intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
        for i in range(10)
    ]
    track = Track(id=1, particles=particles, start_frame=0, end_frame=9)
    
    # Should handle NaN values gracefully
    features = calculator.calculate_track_features(track)
    assert features is None

def test_confinement_ratio(calculator):
    """Test confinement ratio calculation"""
    # Create a confined track (moving back and forth)
    particles = []
    for i in range(10):
        particles.append(
            Particle(
                frame=i,
                x=5.0 * np.sin(i * np.pi/2),  # Oscillating between -5 and 5
                y=0.0,
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    track = Track(id=1, particles=particles, start_frame=0, end_frame=9)
    
    features = calculator.calculate_track_features(track)
    assert features is not None
    # Confinement ratio should be low for confined motion
    assert features.confinement_ratio < 0.3

def _track(xy, tid=9):
    particles = [Particle(frame=i, x=float(x), y=float(y), intensity=100.0, sigma=1.5, snr=10.0,
                          frame_size=(100, 100)) for i, (x, y) in enumerate(xy)]
    return Track(id=tid, particles=particles, start_frame=0, end_frame=len(xy) - 1)


def test_asymmetry_is_a_real_number(calculator, linear_track):
    """Regression: np.linalg.eigvals returned complex128 (written to CSV as '(1+0j)', dropped by the viewers)."""
    a = calculator.calculate_asymmetry(linear_track.positions)
    assert isinstance(a, float) and not isinstance(a, complex)


def test_features_when_a_lag_has_zero_msd(calculator):
    """Regression: a track that returns exactly to earlier positions has MSD 0 at some lag; log(0) made the
    whole feature calculation fail (None) instead of fitting the positive MSD points."""
    xy = [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0)]
    f = calculator.calculate_track_features(_track(xy))
    assert f is not None and np.isfinite(f.alpha) and np.isfinite(f.fractal_dimension)
