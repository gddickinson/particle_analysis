# src/particle_analysis/core/particle_tracking.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.neighbors import KDTree
from ..core.particle_detection import Particle

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Track:
    """Data class for storing particle track information"""
    id: int
    particles: List[Particle]
    start_frame: int
    end_frame: int
    
    @property
    def length(self) -> int:
        """Number of frames in track"""
        return len(self.particles)
    
    @property
    def positions(self) -> np.ndarray:
        """Array of x,y positions"""
        return np.array([(p.x, p.y) for p in self.particles])
    
    @property
    def frames(self) -> np.ndarray:
        """Array of frame numbers"""
        return np.array([p.frame for p in self.particles])
    
    @property
    def intensities(self) -> np.ndarray:
        """Array of particle intensities"""
        return np.array([p.intensity for p in self.particles])

    def to_dict(self) -> Dict:
        """Convert track to dictionary format"""
        return {
            'id': self.id,
            'frames': self.frames.tolist(),
            'x': self.positions[:, 0].tolist(),
            'y': self.positions[:, 1].tolist(),
            'intensities': self.intensities.tolist(),
            'start_frame': self.start_frame,
            'end_frame': self.end_frame
        }

class ParticleTracker:
    """Class for tracking particles across frames"""
    
    def __init__(self, 
                 max_distance: float = 5.0,
                 max_gap: int = 2,
                 min_track_length: int = 3):
        """
        Initialize particle tracker
        
        Parameters
        ----------
        max_distance : float
            Maximum distance a particle can move between frames
        max_gap : int
            Maximum number of frames a particle can disappear
        min_track_length : int
            Minimum number of frames for a valid track
        """
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.min_track_length = min_track_length
        
    def track_particles(self, particles: List[Particle]) -> List[Track]:
        """
        Link particles across frames to form tracks
        
        Parameters
        ----------
        particles : List[Particle]
            List of detected particles from all frames
            
        Returns
        -------
        List[Track]
            List of particle tracks
        """
        try:
            # Sort particles by frame
            particles_by_frame = self._sort_particles_by_frame(particles)
            
            # Initialize tracks
            current_tracks = []
            next_track_id = 0
            
            # Process each frame
            frames = sorted(particles_by_frame.keys())
            for frame in frames:
                current_particles = particles_by_frame[frame]
                
                if not current_tracks:
                    # Initialize tracks with first frame particles
                    for particle in current_particles:
                        track = Track(
                            id=next_track_id,
                            particles=[particle],
                            start_frame=frame,
                            end_frame=frame
                        )
                        current_tracks.append(track)
                        next_track_id += 1
                else:
                    # Link current particles to existing tracks
                    self._link_particles_to_tracks(
                        current_tracks, current_particles, frame
                    )
            
            # Filter tracks by minimum length
            valid_tracks = [
                track for track in current_tracks 
                if len(track.particles) >= self.min_track_length
            ]
            
            # Update track properties
            for track in valid_tracks:
                track.end_frame = track.particles[-1].frame
            
            return valid_tracks
            
        except Exception as e:
            logger.error(f"Error in particle tracking: {str(e)}")
            raise
            
    def _sort_particles_by_frame(self, 
                                particles: List[Particle]
                                ) -> Dict[int, List[Particle]]:
        """Sort particles into dictionary by frame number"""
        particles_by_frame = {}
        for particle in particles:
            if particle.frame not in particles_by_frame:
                particles_by_frame[particle.frame] = []
            particles_by_frame[particle.frame].append(particle)
        return particles_by_frame
    
    def _link_particles_to_tracks(self,
                                current_tracks: List[Track],
                                current_particles: List[Particle],
                                frame: int) -> None:
        """Link current frame particles to existing tracks"""
        # Get active tracks (those that could potentially link to current particles)
        active_tracks = [
            track for track in current_tracks
            if track.end_frame >= frame - self.max_gap
        ]
        
        if not active_tracks or not current_particles:
            return
            
        # Build KDTree for current particles
        current_positions = np.array([
            [p.x, p.y] for p in current_particles
        ])
        tree = KDTree(current_positions)
        
        # Get last known positions of active tracks
        track_positions = np.array([
            [t.particles[-1].x, t.particles[-1].y]
            for t in active_tracks
        ])
        
        # Query tree for nearest neighbors
        distances, indices = tree.query(
            track_positions,
            k=1,
            return_distance=True
        )
        
        # Create assignment pairs based on distance threshold
        assignments = []
        used_particles = set()
        
        for track_idx, (dist, particle_idx) in enumerate(zip(distances, indices)):
            if dist[0] <= self.max_distance:
                assignments.append((track_idx, particle_idx[0], dist[0]))
                
        # Sort assignments by distance
        assignments.sort(key=lambda x: x[2])
        
        # Assign particles to tracks
        for track_idx, particle_idx, _ in assignments:
            if particle_idx not in used_particles:
                track = active_tracks[track_idx]
                particle = current_particles[particle_idx]
                
                # Fill any gaps
                self._fill_track_gaps(track, particle)
                
                # Add particle to track
                track.particles.append(particle)
                track.end_frame = frame
                used_particles.add(particle_idx)
                
        # Create new tracks for unassigned particles
        for i, particle in enumerate(current_particles):
            if i not in used_particles:
                track = Track(
                    id=len(current_tracks),
                    particles=[particle],
                    start_frame=frame,
                    end_frame=frame
                )
                current_tracks.append(track)
    
    def _fill_track_gaps(self, track: Track, next_particle: Particle) -> None:
        """Fill gaps in track with interpolated particles"""
        last_particle = track.particles[-1]
        gap = next_particle.frame - last_particle.frame
        
        if gap > 1:
            # Create interpolated particles
            for frame in range(last_particle.frame + 1, next_particle.frame):
                fraction = (frame - last_particle.frame) / gap
                x = last_particle.x + fraction * (next_particle.x - last_particle.x)
                y = last_particle.y + fraction * (next_particle.y - last_particle.y)
                intensity = last_particle.intensity + fraction * (
                    next_particle.intensity - last_particle.intensity
                )
                
                interpolated = Particle(
                    frame=frame,
                    x=x,
                    y=y,
                    intensity=intensity,
                    sigma=last_particle.sigma,  # Use last known sigma
                    snr=last_particle.snr,      # Use last known SNR
                    frame_size=last_particle.frame_size,
                    id=None
                )
                track.particles.append(interpolated)
    
    def calculate_track_properties(self, track: Track) -> Dict:
        """
        Calculate various track properties
        
        Returns
        -------
        Dict
            Dictionary of track properties including:
            - total_displacement
            - average_speed
            - confinement_ratio
            - etc.
        """
        positions = track.positions
        
        # Calculate displacements
        displacements = np.diff(positions, axis=0)
        step_sizes = np.sqrt(np.sum(displacements**2, axis=1))
        
        # Calculate properties
        total_displacement = np.sqrt(np.sum(
            (positions[-1] - positions[0])**2
        ))
        average_speed = np.mean(step_sizes)
        confinement_ratio = total_displacement / np.sum(step_sizes)
        
        return {
            'total_displacement': total_displacement,
            'average_speed': average_speed,
            'confinement_ratio': confinement_ratio,
            'track_length': track.length,
            'mean_intensity': np.mean(track.intensities)
        }