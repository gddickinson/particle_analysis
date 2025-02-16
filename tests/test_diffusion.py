# tests/test_diffusion.py

import pytest
import numpy as np
from particle_analysis.analysis.diffusion import DiffusionAnalyzer, MotionType
from particle_analysis.core.particle_detection import Particle
from particle_analysis.core.particle_tracking import Track

@pytest.fixture
def analyzer():
    """Create DiffusionAnalyzer instance"""
    return DiffusionAnalyzer(pixel_size=0.108, frame_interval=0.1)

@pytest.fixture
def normal_diffusion_track():
    """Create track with normal diffusion"""
    np.random.seed(42)
    n_steps = 100
    D = 0.1  # Diffusion coefficient
    dt = 0.1  # Time step

    # Generate random walks
    steps = np.random.normal(0, np.sqrt(2*D*dt), (n_steps, 2))
    positions = np.cumsum(steps, axis=0)

    # Create particles
    particles = [
        Particle(frame=i,
                x=float(pos[0]),
                y=float(pos[1]),
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100))
        for i, pos in enumerate(positions)
    ]

    return Track(id=1, particles=particles,
                start_frame=0, end_frame=n_steps-1)

@pytest.fixture
def directed_motion_track():
    """Create track with directed motion"""
    n_steps = 100
    velocity = 1.0  # Velocity
    dt = 0.1  # Time step

    # Generate positions with constant velocity plus noise
    time = np.arange(n_steps) * dt
    positions = np.column_stack([
        velocity * time + np.random.normal(0, 0.1, n_steps),
        np.random.normal(0, 0.1, n_steps)
    ])

    # Create particles
    particles = [
        Particle(frame=i,
                x=float(pos[0]),
                y=float(pos[1]),
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100))
        for i, pos in enumerate(positions)
    ]

    return Track(id=2, particles=particles,
                start_frame=0, end_frame=n_steps-1)

@pytest.fixture
def confined_motion_track():
    """Create track with confined motion"""
    np.random.seed(42)
    n_steps = 100
    confinement_size = 2.0

    # Generate confined random walk
    positions = np.zeros((n_steps, 2))
    current_pos = np.array([0., 0.])

    for i in range(n_steps):
        step = np.random.normal(0, 0.2, 2)
        proposed_pos = current_pos + step

        # Apply confinement
        if np.sum(proposed_pos**2) <= confinement_size**2:
            current_pos = proposed_pos
        positions[i] = current_pos

    # Create particles
    particles = [
        Particle(frame=i,
                x=float(pos[0]),
                y=float(pos[1]),
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100))
        for i, pos in enumerate(positions)
    ]

    return Track(id=3, particles=particles,
                start_frame=0, end_frame=n_steps-1)


def test_msd_calculation(analyzer, normal_diffusion_track):
    """Test MSD calculation"""
    time_lags, msd_values = analyzer.calculate_msd(normal_diffusion_track)

    assert len(time_lags) == len(msd_values)
    assert np.all(time_lags > 0)
    assert np.all(msd_values > 0)
    assert np.all(np.diff(time_lags) > 0)  # Time lags should increase

    # Test with max_points
    time_lags_limited, msd_values_limited = analyzer.calculate_msd(
        normal_diffusion_track, max_points=10
    )
    assert len(time_lags_limited) == 10

def test_diffusion_model_fitting(analyzer, normal_diffusion_track):
    """Test fitting of diffusion models"""
    time_lags, msd_values = analyzer.calculate_msd(normal_diffusion_track)
    fit_results = analyzer.fit_diffusion_models(time_lags, msd_values)

    assert 'normal' in fit_results
    assert 'anomalous' in fit_results
    assert 'confined' in fit_results
    assert 'directed' in fit_results

    # Check normal diffusion fit
    normal_fit = fit_results['normal']
    assert 'D' in normal_fit
    assert normal_fit['D'] > 0
    assert 'r_squared' in normal_fit
    assert 0 <= normal_fit['r_squared'] <= 1

    # Check fit values
    for model in fit_results.values():
        assert len(model['fit_values']) == len(msd_values)

def test_motion_classification(analyzer,
                             normal_diffusion_track,
                             directed_motion_track,
                             confined_motion_track):
    """Test motion type classification"""
    # Test normal diffusion
    motion_type = analyzer.classify_motion(normal_diffusion_track)
    assert isinstance(motion_type, MotionType)
    assert motion_type == MotionType.NORMAL_DIFFUSION

    # Test directed motion
    motion_type = analyzer.classify_motion(directed_motion_track)
    assert motion_type == MotionType.DIRECTED

    # Test confined motion
    motion_type = analyzer.classify_motion(confined_motion_track)
    assert motion_type == MotionType.CONFINED

def test_track_analysis(analyzer, normal_diffusion_track):
    """Test comprehensive track analysis"""
    results = analyzer.analyze_track(normal_diffusion_track)

    assert 'track_id' in results
    assert results['track_id'] == normal_diffusion_track.id

    assert 'motion_type' in results
    assert isinstance(results['motion_type'], MotionType)

    assert 'msd_data' in results
    assert 'time_lags' in results['msd_data']
    assert 'msd_values' in results['msd_data']

    assert 'fit_results' in results
    assert 'instantaneous_diffusion' in results

def test_instantaneous_diffusion(analyzer, normal_diffusion_track):
    """Test instantaneous diffusion coefficient calculation"""
    inst_diff = analyzer.calculate_instantaneous_diffusion(normal_diffusion_track)

    assert len(inst_diff) == len(normal_diffusion_track.particles) - 1
    assert np.all(inst_diff >= 0)  # Should be non-negative
    assert not np.any(np.isnan(inst_diff))  # No NaN values

def test_error_handling(analyzer):
    """Test error handling"""
    # Test with empty track
    empty_track = Track(id=0, particles=[], start_frame=0, end_frame=0)

    # MSD calculation should handle empty track
    time_lags, msd_values = analyzer.calculate_msd(empty_track)
    assert len(time_lags) == 0
    assert len(msd_values) == 0

    # Motion classification should handle errors
    motion_type = analyzer.classify_motion(empty_track)
    assert motion_type == MotionType.NORMAL_DIFFUSION  # Default fallback

    # Track analysis should handle errors
    results = analyzer.analyze_track(empty_track)
    assert isinstance(results, dict)
    assert len(results) == 0

def test_r_squared_calculation(analyzer):
    """Test R-squared calculation"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    r2 = analyzer._r_squared(y_true, y_pred)
    assert 0 <= r2 <= 1

    # Perfect fit should have R2 = 1
    r2_perfect = analyzer._r_squared(y_true, y_true)
    assert np.isclose(r2_perfect, 1.0)

    # Constant prediction should have low R2
    r2_constant = analyzer._r_squared(y_true, np.ones_like(y_true) * np.mean(y_true))
    assert r2_constant < 0.1

def test_pixel_scaling(analyzer, normal_diffusion_track):
    """Test pixel size scaling"""
    # Create analyzer with different pixel size
    analyzer2 = DiffusionAnalyzer(pixel_size=0.216, frame_interval=0.1)

    # Calculate MSD with both analyzers
    _, msd1 = analyzer.calculate_msd(normal_diffusion_track)
    _, msd2 = analyzer2.calculate_msd(normal_diffusion_track)

    # MSD should scale with square of pixel size
    ratio = np.mean(msd2 / msd1)
    expected_ratio = (0.216 / 0.108) ** 2
    assert np.isclose(ratio, expected_ratio, rtol=0.1)

def test_analysis_consistency(analyzer, normal_diffusion_track):
    """Test consistency of analysis results"""
    # Multiple runs should give same results
    results1 = analyzer.analyze_track(normal_diffusion_track)
    results2 = analyzer.analyze_track(normal_diffusion_track)

    assert results1['motion_type'] == results2['motion_type']
    assert np.allclose(
        results1['msd_data']['msd_values'],
        results2['msd_data']['msd_values']
    )

    # Model fits should be consistent
    for model in results1['fit_results']:
        assert np.allclose(
            results1['fit_results'][model]['fit_values'],
            results2['fit_results'][model]['fit_values']
        )
