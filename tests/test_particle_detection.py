# tests/test_particle_detection.py

import pytest
import numpy as np
from particle_analysis.core.particle_detection import ParticleDetector, Particle

@pytest.fixture
def detector():
    """Create a particle detector with default parameters"""
    return ParticleDetector()

@pytest.fixture
def synthetic_frame():
    """Create a synthetic frame with known particles"""
    frame = np.zeros((100, 100))
    # Add Gaussian peaks at known positions
    positions = [(30, 30), (60, 60), (30, 60), (60, 30)]
    for x, y in positions:
        frame[y-2:y+3, x-2:x+3] = np.array([
            [0, 0.5, 1, 0.5, 0],
            [0.5, 1, 2, 1, 0.5],
            [1, 2, 4, 2, 1],
            [0.5, 1, 2, 1, 0.5],
            [0, 0.5, 1, 0.5, 0]
        ])
    # Add noise
    frame += np.random.normal(0, 0.1, frame.shape)
    return frame, positions

def test_particle_dataclass():
    """Test Particle dataclass creation and methods"""
    particle = Particle(
        frame=0,
        x=10.5,
        y=20.5,
        intensity=100.0,
        sigma=1.5,
        snr=10.0,
        frame_size=(100, 100)
    )
    
    assert particle.frame == 0
    assert particle.x == 10.5
    assert particle.y == 20.5
    assert particle.intensity == 100.0
    assert particle.sigma == 1.5
    assert particle.snr == 10.0
    assert particle.frame_size == (100, 100)
    
    # Test to_dict method
    particle_dict = particle.to_dict()
    assert isinstance(particle_dict, dict)
    assert particle_dict['frame'] == 0
    assert particle_dict['x'] == 10.5
    assert particle_dict['y'] == 20.5

def test_detector_initialization():
    """Test detector initialization with different parameters"""
    detector = ParticleDetector(
        min_sigma=1.5,
        max_sigma=4.0,
        sigma_steps=6,
        threshold_rel=0.3,
        min_distance=7
    )
    
    assert detector.min_sigma == 1.5
    assert detector.max_sigma == 4.0
    assert detector.sigma_steps == 6
    assert detector.threshold_rel == 0.3
    assert detector.min_distance == 7
    assert len(detector.sigmas) == 6

def test_gaussian_kernel():
    """Test Gaussian kernel creation"""
    detector = ParticleDetector()
    kernel = detector.create_gaussian_kernel(sigma=1.0, size=5)
    
    assert kernel.shape == (5, 5)
    assert np.allclose(np.sum(kernel), 1.0)
    assert kernel[2, 2] == np.max(kernel)  # Center should be maximum
    
    # Test symmetry
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(kernel[0:2], np.flip(kernel[-2:], axis=0))
    
    # Test invalid size
    with pytest.raises(ValueError):
        detector.create_gaussian_kernel(sigma=1.0, size=4)  # Even size should fail

def test_particle_detection(detector, synthetic_frame):
    """Test particle detection on synthetic data"""
    frame, true_positions = synthetic_frame
    
    # Detect particles
    particles = detector.detect_frame(frame)
    
    # Check number of detected particles
    assert len(particles) == len(true_positions)
    
    # Check if detected positions are close to true positions
    detected_positions = [(p.x, p.y) for p in particles]
    for true_pos in true_positions:
        # Find closest detected position
        min_dist = min(np.sqrt((x - true_pos[0])**2 + (y - true_pos[1])**2) 
                      for x, y in detected_positions)
        assert min_dist < 2.0  # Should be within 2 pixels

def test_movie_detection(detector):
    """Test particle detection on movie data"""
    # Create synthetic movie
    movie = np.zeros((5, 100, 100))
    positions = [(30, 30), (60, 60)]
    
    for frame in range(5):
        for x, y in positions:
            movie[frame, y-2:y+3, x-2:x+3] = np.array([
                [0, 0.5, 1, 0.5, 0],
                [0.5, 1, 2, 1, 0.5],
                [1, 2, 4, 2, 1],
                [0.5, 1, 2, 1, 0.5],
                [0, 0.5, 1, 0.5, 0]
            ])
        movie[frame] += np.random.normal(0, 0.1, (100, 100))
    
    # Detect particles
    particles = detector.detect_movie(movie)
    
    # Check basic properties
    assert len(particles) == len(positions) * 5  # 2 particles × 5 frames
    assert all(0 <= p.frame < 5 for p in particles)
    assert all(isinstance(p, Particle) for p in particles)

def test_error_handling():
    """Test error handling in particle detection"""
    detector = ParticleDetector()
    
    # Test invalid frame dimensions
    with pytest.raises(ValueError):
        detector.detect_frame(np.zeros((10, 10, 10)))
    
    # Test invalid movie dimensions
    with pytest.raises(ValueError):
        detector.detect_movie(np.zeros((10, 10)))
    
    # Test empty frame
    particles = detector.detect_frame(np.zeros((10, 10)))
    assert len(particles) == 0

def test_particle_fitting_edge_cases(detector):
    """Test particle detection in edge cases"""
    # Test frame with single bright pixel
    frame = np.zeros((20, 20))
    frame[10, 10] = 10.0
    particles = detector.detect_frame(frame)
    assert len(particles) > 0
    
    # Test frame with uniform intensity
    frame = np.ones((20, 20))
    particles = detector.detect_frame(frame)
    assert len(particles) == 0
    
    # Test frame with high noise
    frame = np.random.normal(0, 1, (20, 20))
    particles = detector.detect_frame(frame)
    # Number of detected particles should be reasonable
    assert len(particles) < 10

def _spot_frame(sigma, size=41, amp=200.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    c = size // 2 + 0.3
    return 100 + amp * np.exp(-((x - c)**2 + (y - c)**2) / (2 * sigma**2)) + rng.normal(0, 2, (size, size))


def test_sigma_range_bounds_the_fitted_width():
    """Regression: min_sigma/max_sigma were ignored (fit bounds were fixed at 0.8-4 px)."""
    frame = _spot_frame(2.5)
    wide = ParticleDetector(min_sigma=1.0, max_sigma=4.0, min_distance=5).detect_frame(frame)
    assert len(wide) == 1 and abs(wide[0].sigma - 2.5) < 0.1
    narrow = ParticleDetector(min_sigma=1.0, max_sigma=2.0, min_distance=5).detect_frame(frame)
    assert all(p.sigma <= 2.0 + 1e-9 for p in narrow)
    big = ParticleDetector(min_sigma=3.0, max_sigma=4.0, min_distance=5).detect_frame(_spot_frame(1.5))
    assert all(p.sigma >= 3.0 - 1e-9 for p in big)


def test_sigma_steps_sets_the_trial_widths():
    det = ParticleDetector(min_sigma=1.0, max_sigma=3.0, sigma_steps=3)
    assert np.allclose(det.sigmas, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ParticleDetector(min_sigma=3.0, max_sigma=1.0)
    with pytest.raises(ValueError):
        ParticleDetector(sigma_steps=0)
