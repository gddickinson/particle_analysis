# src/particle_analysis/io/readers.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import logging
import skimage.io as skio
from ..core.particle_detection import Particle
from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures

# Set up logging
logger = logging.getLogger(__name__)

class DataReader:
    """Class for reading various data formats used in particle analysis"""
    
    def __init__(self, pixel_size: float = 0.108):
        """
        Initialize reader with microscope parameters
        
        Parameters
        ----------
        pixel_size : float
            Pixel size in micrometers
        """
        self.pixel_size = pixel_size
        
    def read_movie(self, 
                   file_path: Union[str, Path],
                   channel: int = 0) -> Optional[np.ndarray]:
        """
        Read movie file (TIFF stack)
        
        Parameters
        ----------
        file_path : str or Path
            Path to movie file
        channel : int
            Channel to read for multi-channel files
            
        Returns
        -------
        np.ndarray
            Movie data as (t, y, x) array
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read movie file
            movie = skio.imread(str(file_path))
            
            # Handle different dimensionalities
            if movie.ndim == 3:
                # Already in (t, y, x) format
                return movie
            elif movie.ndim == 4:
                # Multi-channel format (t, c, y, x)
                return movie[:, channel, :, :]
            else:
                raise ValueError(f"Unexpected movie dimensions: {movie.ndim}")
                
        except Exception as e:
            logger.error(f"Error reading movie file {file_path}: {str(e)}")
            return None
            
    def read_particles_csv(self, 
                          file_path: Union[str, Path]) -> Optional[List[Particle]]:
        """
        Read particle detections from CSV file
        
        Parameters
        ----------
        file_path : str or Path
            Path to CSV file containing particle data
            
        Returns
        -------
        List[Particle]
            List of detected particles
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['frame', 'x', 'y', 'intensity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Convert to particles
            particles = []
            for _, row in df.iterrows():
                particle = Particle(
                    frame=int(row['frame']),
                    x=float(row['x']),
                    y=float(row['y']),
                    intensity=float(row['intensity']),
                    sigma=float(row.get('sigma', 1.5)),
                    snr=float(row.get('snr', 0.0)),
                    frame_size=None,  # Will be set when processing
                    id=int(row['id']) if 'id' in row else None
                )
                particles.append(particle)
                
            return particles
            
        except Exception as e:
            logger.error(f"Error reading particles file {file_path}: {str(e)}")
            return None
            
    def read_tracks_csv(self, 
                       file_path: Union[str, Path]) -> Optional[List[Track]]:
        """
        Read particle tracks from CSV file
        
        Parameters
        ----------
        file_path : str or Path
            Path to CSV file containing track data
            
        Returns
        -------
        List[Track]
            List of particle tracks
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['track_number', 'frame', 'x', 'y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Group by track number and create Track objects
            tracks = []
            for track_id, group in df.groupby('track_number'):
                group = group.sort_values('frame')
                
                # Create particles for this track
                particles = []
                for _, row in group.iterrows():
                    particle = Particle(
                        frame=int(row['frame']),
                        x=float(row['x']),
                        y=float(row['y']),
                        intensity=float(row.get('intensity', 0.0)),
                        sigma=float(row.get('sigma', 1.5)),
                        snr=float(row.get('snr', 0.0)),
                        frame_size=None,
                        id=int(row['id']) if 'id' in row else None
                    )
                    particles.append(particle)
                
                track = Track(
                    id=int(track_id),
                    particles=particles,
                    start_frame=int(group['frame'].min()),
                    end_frame=int(group['frame'].max())
                )
                tracks.append(track)
                
            return tracks
            
        except Exception as e:
            logger.error(f"Error reading tracks file {file_path}: {str(e)}")
            return None
            
    def read_features_csv(self, 
                         file_path: Union[str, Path]
                         ) -> Optional[List[TrackFeatures]]:
        """
        Read track features from CSV file
        
        Parameters
        ----------
        file_path : str or Path
            Path to CSV file containing feature data
            
        Returns
        -------
        List[TrackFeatures]
            List of track features
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Convert to TrackFeatures objects
            features = []
            for _, row in df.iterrows():
                feature = TrackFeatures(
                    track_id=int(row['track_id']),
                    msd_values=np.array(eval(row['msd_values'])),
                    diffusion_coefficient=float(row['diffusion_coefficient']),
                    alpha=float(row['alpha']),
                    radius_gyration=float(row['radius_gyration']),
                    asymmetry=float(row['asymmetry']),
                    fractal_dimension=float(row['fractal_dimension']),
                    straightness=float(row['straightness']),
                    mean_velocity=float(row['mean_velocity']),
                    mean_acceleration=float(row['mean_acceleration']),
                    confinement_ratio=float(row['confinement_ratio'])
                )
                features.append(feature)
                
            return features
            
        except Exception as e:
            logger.error(f"Error reading features file {file_path}: {str(e)}")
            return None