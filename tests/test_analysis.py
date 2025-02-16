# tests/test_analysis.py

import pytest
import numpy as np
from particle_analysis.analysis.statistics import TrackAnalyzer
from particle_analysis.core.particle_detection import Particle
from particle_analysis.core.particle_tracking import Track
from particle_analysis.core.feature_calculation import TrackFeatures
import tempfile
from pathlib import Path

@pytest.fixture
def analyzer():
    """Create TrackAnalyzer instance"""
    return TrackAnalyzer()

@pytest.fixture
def sample_tracks():
    """Create sample tracks for testing"""
    tracks = []

    # Create linear track
    particles1 = [
        Particle(frame=i, x=float(i), y=0.0, intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
        for i in range(10)
    ]
    tracks.append(Track(id=1, particles=particles1,
                       start_frame=0, end_frame=9))

    # Create circular track
    particles2 = [
        Particle(frame=i,
                x=10*np.cos(2*np.pi*i/20),
                y=10*np.sin(2*np.pi*i/20),
                intensity=120.0, sigma=1.5, snr=12.0,
                frame_size=(100, 100))
        for i in range(20)
    ]
    tracks.append(Track(id=2, particles=particles2,
                       start_frame=0, end_frame=19))

    return tracks

@pytest.fixture
def sample_features():
    """Create sample features for testing"""
    features = []

    # Linear motion features
    features.append(TrackFeatures(
        track_id=1,
        msd_values=np.array([1.0, 4.0, 9.0]),
        diffusion_coefficient=0.5,
        alpha=2.0,  # Ballistic
        radius_gyration=2.0,
        asymmetry=0.1,
        fractal_dimension=1.1,
        straightness=0.9,
        mean_velocity=1.0,
        mean_acceleration=0.1,
        confinement_ratio=0.9
    ))

    # Circular motion features
    features.append(TrackFeatures(
        track_id=2,
        msd_values=np.array([1.0, 2.0, 3.0]),
        diffusion_coefficient=0.2,
        alpha=1.0,  # Diffusive
        radius_gyration=10.0,
        asymmetry=0.05,
        fractal_dimension=1.8,
        straightness=0.1,
        mean_velocity=2.0,
        mean_acceleration=0.2,
        confinement_ratio=0.2
    ))

    return features


@pytest.fixture
def condition_features():
    """Create features for two conditions"""
    condition1 = []
    condition2 = []

    # Condition 1: More directed motion
    for i in range(5):
        condition1.append(TrackFeatures(
            track_id=i,
            msd_values=np.array([1.0, 4.0, 9.0]) + np.random.normal(0, 0.1, 3),
            diffusion_coefficient=0.5 + np.random.normal(0, 0.05),
            alpha=1.8 + np.random.normal(0, 0.1),
            radius_gyration=2.0 + np.random.normal(0, 0.2),
            asymmetry=0.1 + np.random.normal(0, 0.01),
            fractal_dimension=1.1 + np.random.normal(0, 0.05),
            straightness=0.9 + np.random.normal(0, 0.05),
            mean_velocity=1.0 + np.random.normal(0, 0.1),
            mean_acceleration=0.1 + np.random.normal(0, 0.01),
            confinement_ratio=0.9 + np.random.normal(0, 0.05)
        ))

    # Condition 2: More diffusive motion
    for i in range(5):
        condition2.append(TrackFeatures(
            track_id=i+5,
            msd_values=np.array([1.0, 2.0, 3.0]) + np.random.normal(0, 0.1, 3),
            diffusion_coefficient=0.2 + np.random.normal(0, 0.05),
            alpha=1.0 + np.random.normal(0, 0.1),
            radius_gyration=5.0 + np.random.normal(0, 0.2),
            asymmetry=0.5 + np.random.normal(0, 0.05),
            fractal_dimension=1.8 + np.random.normal(0, 0.05),
            straightness=0.3 + np.random.normal(0, 0.05),
            mean_velocity=0.5 + np.random.normal(0, 0.1),
            mean_acceleration=0.05 + np.random.normal(0, 0.01),
            confinement_ratio=0.3 + np.random.normal(0, 0.05)
        ))

    return condition1, condition2

def test_track_classification(analyzer, sample_features):
    """Test track classification"""
    labels = analyzer.classify_tracks(sample_features, n_classes=2)

    assert len(labels) == len(sample_features)
    assert len(np.unique(labels)) == 2
    assert labels[0] != labels[1]  # Different classes for different motions

def test_condition_comparison(analyzer, condition_features):
    """Test statistical comparison between conditions"""
    condition1, condition2 = condition_features
    results = analyzer.compare_conditions(condition1, condition2)

    assert isinstance(results, dict)
    assert len(results) > 0

    # Check statistical test results
    for feature, stats in results.items():
        assert 't_pvalue' in stats
        assert 'u_pvalue' in stats
        assert 'cohens_d' in stats
        # Should have significant differences
        assert stats['t_pvalue'] < 0.05 or stats['u_pvalue'] < 0.05

def test_mobility_analysis(analyzer, sample_tracks, sample_features):
    """Test mobility analysis"""
    results = analyzer.analyze_mobility(sample_tracks, sample_features)

    assert 'instant_velocities' in results
    assert 'turning_angles' in results
    assert 'confinement_ratios' in results

    # Check values for different motion types
    velocities = results['instant_velocities']
    angles = results['turning_angles']

    # Linear track should have constant velocity
    assert np.std(velocities[:9]) < 0.1
    # Circular track should have angles around ±18 degrees
    assert np.abs(np.mean(angles[9:])) - 18.0 < 2.0

def test_spatial_statistics(analyzer, sample_tracks):
    """Test spatial statistics calculation"""
    results = analyzer.calculate_spatial_statistics(
        sample_tracks,
        frame_size=(100, 100),
        bin_size=10
    )

    assert 'density_map' in results
    assert 'velocity_map' in results

    density_map = results['density_map']
    velocity_map = results['velocity_map']

    assert density_map.shape == (10, 10)
    assert velocity_map.shape == (10, 10)
    assert np.sum(density_map) > 0
    assert not np.all(np.isnan(velocity_map))

def test_report_generation(analyzer, sample_tracks, sample_features):
    """Test analysis report generation"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname) / "analysis_report.xlsx"
        success = analyzer.generate_report(
            sample_tracks,
            sample_features,
            str(output_path)
        )

        assert success
        assert output_path.exists()

        # Check file content
        import pandas as pd
        summary = pd.read_excel(output_path, sheet_name='Summary')
        assert len(summary) > 0
        assert 'n_tracks' in summary.columns

        classification = pd.read_excel(output_path, sheet_name='Classification')
        assert len(classification) > 0

        mobility = pd.read_excel(output_path, sheet_name='Mobility')
        assert len(mobility) > 0

def test_error_handling(analyzer):
    """Test error handling in analysis"""
    # Test with empty data
    labels = analyzer.classify_tracks([])
    assert len(labels) == 0

    # Test with invalid features
    results = analyzer.compare_conditions([], [])
    assert len(results) == 0

    # Test with invalid track data
    results = analyzer.analyze_mobility([], [])
    assert all(len(v) == 0 for v in results.values())

def test_feature_matrix_creation(analyzer, sample_features):
    """Test feature matrix creation"""
    matrix = analyzer._create_feature_matrix(sample_features)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == len(sample_features)
    assert matrix.shape[1] > 0  # Should have multiple features
    assert not np.any(np.isnan(matrix))  # No NaN values
    assert not np.any(np.isinf(matrix))  # No infinite values

def test_analysis_parameters(analyzer, sample_features):
    """Test analysis with different parameters"""
    # Test classification with different number of classes
    labels_2 = analyzer.classify_tracks(sample_features, n_classes=2)
    labels_3 = analyzer.classify_tracks(sample_features, n_classes=3)

    assert len(np.unique(labels_2)) == 2
    assert len(np.unique(labels_3)) == 3

    # Test mobility analysis with different time windows
    results_5 = analyzer.analyze_mobility(
        [sample_features[0]],
        sample_features,
        time_window=5
    )
    results_10 = analyzer.analyze_mobility(
        [sample_features[0]],
        sample_features,
        time_window=10
    )

    assert len(results_5['confinement_ratios']) > len(results_10['confinement_ratios'])

def test_analysis_consistency(analyzer, sample_features):
    """Test consistency of analysis results"""
    # Multiple runs should give same results
    labels1 = analyzer.classify_tracks(sample_features)
    labels2 = analyzer.classify_tracks(sample_features)

    assert np.array_equal(labels1, labels2)

    # Results should be deterministic
    comparison1 = analyzer.compare_conditions(
        sample_features[:1],
        sample_features[1:]
    )
    comparison2 = analyzer.compare_conditions(
        sample_features[:1],
        sample_features[1:]
    )

    for feature in comparison1:
        assert comparison1[feature]['t_pvalue'] == comparison2[feature]['t_pvalue']
