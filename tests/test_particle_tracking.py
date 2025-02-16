# tests/test_particle_tracking.py

import pytest
import numpy as np
from particle_analysis.core.particle_tracking import ParticleTracker, Track
from particle_analysis.core.particle_detection import Particle

@pytest.fixture
def sample_particles():
    """Create a list of sample particles for testing"""
    particles = []
    # Create particles moving in a straight line across frames
    for frame in range(5):
        particles.append(
            Particle(
                frame=frame,
                x=10.0 + frame * 2,  # Moving right
                y=15.0,              # Constant y
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
        # Add another particle moving diagonally
        particles.append(
            Particle(
                frame=frame,
                x=50.0 + frame,      # Moving right slower
                y=50.0 + frame,      # Moving up
                intensity=120.0,
                sigma=1.5,
                snr=12.0,
                frame_size=(100, 100)
            )
        )
    return particles

@pytest.fixture
def tracker():
    """Create a particle tracker with default parameters"""
    return ParticleTracker(max_distance=5.0, max_gap=2, min_track_length=3)

def test_track_dataclass():
    """Test Track dataclass creation and methods"""
    particles = [
        Particle(frame=i, x=float(i), y=2.0, intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
        for i in range(5)
    ]
    
    track = Track(
        id=1,
        particles=particles,
        start_frame=0,
        end_frame=4
    )
    
    assert track.id == 1
    assert track.length == 5
    assert track.start_frame == 0
    assert track.end_frame == 4
    
    # Test properties
    assert len(track.positions) == 5
    assert len(track.frames) == 5
    assert len(track.intensities) == 5
    
    # Test to_dict method
    track_dict = track.to_dict()
    assert isinstance(track_dict, dict)
    assert track_dict['id'] == 1
    assert len(track_dict['frames']) == 5
    assert len(track_dict['x']) == 5
    assert len(track_dict['y']) == 5

def test_particle_tracking(tracker, sample_particles):
    """Test basic particle tracking functionality"""
    tracks = tracker.track_particles(sample_particles)
    
    # Should find 2 tracks
    assert len(tracks) == 2
    
    # Each track should have 5 particles
    for track in tracks:
        assert track.length == 5
        assert track.start_frame == 0
        assert track.end_frame == 4
        
    # Check track properties
    track_props = tracker.calculate_track_properties(tracks[0])
    assert 'total_displacement' in track_props
    assert 'average_speed' in track_props
    assert 'confinement_ratio' in track_props

def test_gap_handling():
    """Test tracking with missing frames"""
    tracker = ParticleTracker(max_distance=5.0, max_gap=2)
    
    # Create particles with a gap
    particles = []
    for frame in [0, 1, 3, 4]:  # Skip frame 2
        particles.append(
            Particle(
                frame=frame,
                x=10.0 + frame * 2,
                y=15.0,
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    
    tracks = tracker.track_particles(particles)
    assert len(tracks) == 1
    assert tracks[0].length == 5  # Should include interpolated particle
    
    # Check interpolated particle
    track_frames = [p.frame for p in tracks[0].particles]
    assert 2 in track_frames  # Frame 2 should be present
    
def test_distance_threshold():
    """Test maximum distance threshold"""
    tracker = ParticleTracker(max_distance=2.0)  # Small max distance
    
    particles = []
    # Create two particles moving too far apart
    particles.append(
        Particle(frame=0, x=10.0, y=10.0, intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
    )
    particles.append(
        Particle(frame=1, x=15.0, y=10.0, intensity=100.0,  # Moves 5 pixels
                sigma=1.5, snr=10.0, frame_size=(100, 100))
    )
    
    tracks = tracker.track_particles(particles)
    assert len(tracks) == 0  # No tracks should meet minimum length requirement

def test_minimum_track_length():
    """Test minimum track length filtering"""
    tracker = ParticleTracker(min_track_length=4)
    
    particles = []
    # Create track with only 3 particles
    for frame in range(3):
        particles.append(
            Particle(
                frame=frame,
                x=10.0,
                y=10.0,
                intensity=100.0,
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    
    tracks = tracker.track_particles(particles)
    assert len(tracks) == 0  # Track should be filtered out due to length

def test_track_properties():
    """Test calculation of track properties"""
    particles = []
    # Create straight line motion
    for frame in range(5):
        particles.append(
            Particle(
                frame=frame,
                x=10.0 + frame,  # Moving at 1 pixel per frame
                y=10.0,          # Constant y
                intensity=100.0 + frame * 10,  # Increasing intensity
                sigma=1.5,
                snr=10.0,
                frame_size=(100, 100)
            )
        )
    
    tracker = ParticleTracker()
    tracks = tracker.track_particles(particles)
    assert len(tracks) == 1
    
    properties = tracker.calculate_track_properties(tracks[0])
    
    # Check specific property values
    assert np.isclose(properties['total_displacement'], 4.0)  # Total displacement should be 4
    assert np.isclose(properties['average_speed'], 1.0)      # Speed should be 1 pixel per frame
    assert np.isclose(properties['confinement_ratio'], 1.0)  # Straight line should have ratio 1
    assert properties['track_length'] == 5
    assert np.isclose(properties['mean_intensity'], 120.0)   # Average of 100,110,120,130,140

def test_error_handling():
    """Test error handling in tracking"""
    tracker = ParticleTracker()
    
    # Test with empty particle list
    tracks = tracker.track_particles([])
    assert len(tracks) == 0
    
    # Test with invalid frame numbers
    particles = [
        Particle(frame=-1, x=0.0, y=0.0, intensity=100.0,
                sigma=1.5, snr=10.0, frame_size=(100, 100))
    ]
    tracks = tracker.track_particles(particles)
    assert len(tracks) == 0

def test_multiple_nearby_particles():
    """Test tracking when multiple particles are nearby"""
    particles = []
    # Two particles moving close to each other
    for frame in range(3):
        particles.extend([
            Particle(frame=frame, x=10.0 + frame, y=10.0, intensity=100.0,
                    sigma=1.5, snr=10.0, frame_size=(100, 100)),
            Particle(frame=frame, x=11.0 + frame, y=10.0, intensity=100.0,
                    sigma=1.5, snr=10.0, frame_size=(100, 100))
        ])
    
    tracker = ParticleTracker(max_distance=2.0)
    tracks = tracker.track_particles(particles)
    
    assert len(tracks) == 2  # Should maintain two separate tracks
    # Check that tracks remain separate
    track1_x = tracks[0].positions[:, 0]
    track2_x = tracks[1].positions[:, 0]
    assert np.all(np.abs(track1_x - track2_x) >= 1.0)  # Should maintain separation