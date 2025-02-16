# tests/test_io.py

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from particle_analysis.io.readers import DataReader
from particle_analysis.io.writers import DataWriter
from particle_analysis.core.particle_detection import Particle
from particle_analysis.core.particle_tracking import Track
from particle_analysis.core.feature_calculation import TrackFeatures

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def sample_particles():
    """Create sample particle data"""
    return [
        Particle(frame=0, x=1.0, y=2.0, intensity=100.0, sigma=1.5, snr=10.0,
                frame_size=(100, 100), id=1),
        Particle(frame=0, x=5.0, y=6.0, intensity=120.0, sigma=1.5, snr=12.0,
                frame_size=(100, 100), id=2),
        Particle(frame=1, x=1.5, y=2.2, intensity=95.0, sigma=1.5, snr=9.5,
                frame_size=(100, 100), id=3)
    ]

@pytest.fixture
def sample_tracks(sample_particles):
    """Create sample track data"""
    return [
        Track(id=1, 
              particles=sample_particles[:2],
              start_frame=0,
              end_frame=0),
        Track(id=2,
              particles=[sample_particles[2]],
              start_frame=1,
              end_frame=1)
    ]

@pytest.fixture
def sample_features():
    """Create sample feature data"""
    return [
        TrackFeatures(
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
        ),
        TrackFeatures(
            track_id=2,
            msd_values=np.array([2.0, 4.0, 6.0]),
            diffusion_coefficient=1.0,
            alpha=0.8,
            radius_gyration=3.0,
            asymmetry=0.2,
            fractal_dimension=1.8,
            straightness=0.7,
            mean_velocity=2.0,
            mean_acceleration=0.2,
            confinement_ratio=0.6
        )
    ]

def test_particle_io(temp_dir, sample_particles):
    """Test writing and reading particles"""
    writer = DataWriter()
    reader = DataReader()
    
    # Write particles
    file_path = Path(temp_dir) / "particles.csv"
    assert writer.write_particles_csv(sample_particles, file_path)
    
    # Read particles back
    loaded_particles = reader.read_particles_csv(file_path)
    assert loaded_particles is not None
    assert len(loaded_particles) == len(sample_particles)
    
    # Compare original and loaded particles
    for orig, loaded in zip(sample_particles, loaded_particles):
        assert orig.frame == loaded.frame
        assert orig.x == loaded.x
        assert orig.y == loaded.y
        assert orig.intensity == loaded.intensity
        assert orig.id == loaded.id

def test_track_io(temp_dir, sample_tracks):
    """Test writing and reading tracks"""
    writer = DataWriter()
    reader = DataReader()
    
    # Write tracks
    file_path = Path(temp_dir) / "tracks.csv"
    assert writer.write_tracks_csv(sample_tracks, file_path)
    
    # Read tracks back
    loaded_tracks = reader.read_tracks_csv(file_path)
    assert loaded_tracks is not None
    assert len(loaded_tracks) == len(sample_tracks)
    
    # Compare original and loaded tracks
    for orig, loaded in zip(sample_tracks, loaded_tracks):
        assert orig.id == loaded.id
        assert orig.start_frame == loaded.start_frame
        assert orig.end_frame == loaded.end_frame
        assert len(orig.particles) == len(loaded.particles)

def test_feature_io(temp_dir, sample_features):
    """Test writing and reading features"""
    writer = DataWriter()
    reader = DataReader()
    
    # Write features
    file_path = Path(temp_dir) / "features.csv"
    assert writer.write_features_csv(sample_features, file_path)
    
    # Read features back
    loaded_features = reader.read_features_csv(file_path)
    assert loaded_features is not None
    assert len(loaded_features) == len(sample_features)
    
    # Compare original and loaded features
    for orig, loaded in zip(sample_features, loaded_features):
        assert orig.track_id == loaded.track_id
        assert np.allclose(orig.msd_values, loaded.msd_values)
        assert orig.diffusion_coefficient == loaded.diffusion_coefficient
        assert orig.alpha == loaded.alpha
        assert orig.radius_gyration == loaded.radius_gyration

def test_analysis_summary(temp_dir, sample_tracks, sample_features):
    """Test writing analysis summary"""
    writer = DataWriter()
    
    # Write summary
    file_path = Path(temp_dir) / "summary.xlsx"
    assert writer.write_analysis_summary(sample_tracks, sample_features, file_path)
    
    # Verify file exists and can be read
    assert file_path.exists()
    df_tracks = pd.read_excel(file_path, sheet_name='Tracks')
    df_features = pd.read_excel(file_path, sheet_name='Features')
    df_summary = pd.read_excel(file_path, sheet_name='Summary')
    
    assert len(df_tracks) == len(sample_tracks)
    assert len(df_features) == len(sample_features)
    assert len(df_summary) == 1

def test_batch_results(temp_dir):
    """Test writing batch processing results"""
    writer = DataWriter()
    
    # Create sample batch results
    results = {
        'file1.tif': {
            'n_tracks': 10,
            'n_particles': 100,
            'mean_track_length': 10.0,
            'mean_diffusion_coef': 0.5,
            'processing_time': 5.0,
            'status': 'success'
        },
        'file2.tif': {
            'n_tracks': 15,
            'n_particles': 150,
            'mean_track_length': 12.0,
            'mean_diffusion_coef': 0.6,
            'processing_time': 6.0,
            'status': 'success'
        }
    }
    
    # Write batch results
    file_path = Path(temp_dir) / "batch_results.xlsx"
    assert writer.write_batch_results(results, file_path)
    
    # Verify file exists and can be read
    assert file_path.exists()
    df_summary = pd.read_excel(file_path, sheet_name='Batch Summary')
    df_stats = pd.read_excel(file_path, sheet_name='Aggregate Stats')
    
    assert len(df_summary) == len(results)
    assert len(df_stats) == 1
    assert df_stats['total_files'].iloc[0] == len(results)

def test_error_handling(temp_dir):
    """Test error handling in IO operations"""
    writer = DataWriter()
    reader = DataReader()
    
    # Test reading non-existent file
    assert reader.read_particles_csv("nonexistent.csv") is None
    
    # Test writing to invalid path
    assert not writer.write_particles_csv([], "/invalid/path/file.csv")
    
    # Test reading invalid CSV format
    invalid_csv = Path(temp_dir) / "invalid.csv"
    invalid_csv.write_text("invalid,csv,format")
    assert reader.read_particles_csv(invalid_csv) is None

def test_movie_reading():
    """Test movie file reading"""
    reader = DataReader()
    
    # Create dummy movie file
    movie = np.random.rand(10, 100, 100)  # (t, y, x)
    with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
        import tifffile
        tifffile.imwrite(tmp.name, movie)
        
        # Read movie
        loaded_movie = reader.read_movie(tmp.name)
        assert loaded_movie is not None
        assert loaded_movie.shape == movie.shape
        assert np.allclose(loaded_movie, movie)