# src/particle_analysis/core/feature_calculation.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist
from ..core.particle_tracking import Track
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class TrackFeatures:
    """Data class for storing calculated track features"""
    track_id: int
    msd_values: np.ndarray
    diffusion_coefficient: float
    alpha: float  # Anomalous diffusion exponent
    radius_gyration: float
    asymmetry: float
    fractal_dimension: float
    straightness: float
    mean_velocity: float
    mean_acceleration: float
    confinement_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert features to dictionary format"""
        return {
            'track_id': self.track_id,
            'msd_values': self.msd_values.tolist(),
            'diffusion_coefficient': self.diffusion_coefficient,
            'alpha': self.alpha,
            'radius_gyration': self.radius_gyration,
            'asymmetry': self.asymmetry,
            'fractal_dimension': self.fractal_dimension,
            'straightness': self.straightness,
            'mean_velocity': self.mean_velocity,
            'mean_acceleration': self.mean_acceleration,
            'confinement_ratio': self.confinement_ratio
        }

class FeatureCalculator:
    """Class for calculating various track features"""
    
    def __init__(self, 
                 max_msd_points: int = 10,
                 min_track_length: int = 5):
        """
        Initialize feature calculator
        
        Parameters
        ----------
        max_msd_points : int
            Maximum number of points to use in MSD calculation
        min_track_length : int
            Minimum track length for feature calculation
        """
        self.max_msd_points = max_msd_points
        self.min_track_length = min_track_length
        
    def calculate_track_features(self, track: Track) -> Optional[TrackFeatures]:
        """
        Calculate features for a single track
        
        Parameters
        ----------
        track : Track
            Track to analyze
            
        Returns
        -------
        Optional[TrackFeatures]
            Calculated features or None if track is too short
        """
        try:
            if len(track.particles) < self.min_track_length:
                return None
                
            positions = track.positions
            times = track.frames
            
            # Calculate basic features
            msd_values, time_lags = self.calculate_msd(positions, times)
            diff_coef, alpha = self.fit_msd_curve(msd_values, time_lags)
            
            # Calculate shape features
            rg = self.calculate_radius_gyration(positions)
            asymmetry = self.calculate_asymmetry(positions)
            fractal_dim = self.calculate_fractal_dimension(positions)
            
            # Calculate dynamic features
            straightness = self.calculate_straightness(positions)
            mean_vel, mean_acc = self.calculate_dynamics(positions, times)
            conf_ratio = self.calculate_confinement_ratio(positions)
            
            return TrackFeatures(
                track_id=track.id,
                msd_values=msd_values,
                diffusion_coefficient=diff_coef,
                alpha=alpha,
                radius_gyration=rg,
                asymmetry=asymmetry,
                fractal_dimension=fractal_dim,
                straightness=straightness,
                mean_velocity=mean_vel,
                mean_acceleration=mean_acc,
                confinement_ratio=conf_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating features for track {track.id}: {str(e)}")
            return None
    
    def calculate_msd(self, 
                     positions: np.ndarray,
                     times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Mean Square Displacement curve"""
        n_points = len(positions)
        max_lag = min(self.max_msd_points, n_points - 1)
        
        msd_values = np.zeros(max_lag)
        time_lags = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            # Calculate displacements for this lag
            disp = positions[lag:] - positions[:-lag]
            # Calculate squared displacements
            squared_disp = np.sum(disp**2, axis=1)
            # Average over all pairs
            msd_values[lag-1] = np.mean(squared_disp)
            # Store corresponding time lag
            time_lags[lag-1] = np.mean(times[lag:] - times[:-lag])
            
        return msd_values, time_lags
    
    def fit_msd_curve(self,
                     msd_values: np.ndarray,
                     time_lags: np.ndarray) -> Tuple[float, float]:
        """Fit MSD curve to extract diffusion coefficient and alpha"""
        # Take log of both arrays
        log_msd = np.log(msd_values)
        log_time = np.log(time_lags)
        
        # Perform linear fit
        slope, intercept, _, _, _ = stats.linregress(log_time, log_msd)
        
        # Extract parameters
        alpha = slope
        diff_coef = np.exp(intercept) / (4 * (alpha if alpha != 0 else 1))
        
        return diff_coef, alpha
    
    def calculate_radius_gyration(self, positions: np.ndarray) -> float:
        """Calculate radius of gyration"""
        center = np.mean(positions, axis=0)
        rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
        return rg
    
    def calculate_asymmetry(self, positions: np.ndarray) -> float:
        """Calculate asymmetry based on principal components"""
        # Calculate covariance matrix
        cov_matrix = np.cov(positions.T)
        
        # Get eigenvalues
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals.sort()
        
        # Calculate asymmetry
        if len(eigenvals) > 1 and np.sum(eigenvals) > 0:
            asymmetry = 1 - (eigenvals[0] / eigenvals[-1])
        else:
            asymmetry = 0
            
        return asymmetry
    
    def calculate_fractal_dimension(self, positions: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        # Calculate pairwise distances
        distances = pdist(positions)
        
        if len(distances) == 0:
            return 0
            
        # Use range of scales
        scales = np.logspace(np.log10(min(distances)/2),
                           np.log10(max(distances)*2), 20)
        counts = []
        
        # Count boxes at each scale
        for scale in scales:
            # Normalize positions to current scale
            scaled_pos = positions / scale
            # Round to get box indices
            boxes = np.ascontiguousarray(np.floor(scaled_pos)).view(
                np.dtype((np.void, scaled_pos.dtype.itemsize * scaled_pos.shape[1]))
            )
            # Count unique boxes
            n_boxes = len(np.unique(boxes))
            counts.append(n_boxes)
            
        # Perform linear fit in log-log space
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]
        
        return fractal_dim
    
    def calculate_straightness(self, positions: np.ndarray) -> float:
        """Calculate track straightness"""
        if len(positions) < 2:
            return 0
            
        # Calculate total path length
        segment_lengths = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_length = np.sum(segment_lengths)
        
        # Calculate end-to-end distance
        end_to_end = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        
        # Calculate straightness
        if total_length > 0:
            straightness = end_to_end / total_length
        else:
            straightness = 0
            
        return straightness
    
    def calculate_dynamics(self, 
                         positions: np.ndarray,
                         times: np.ndarray) -> Tuple[float, float]:
        """Calculate mean velocity and acceleration"""
        if len(positions) < 2:
            return 0, 0
            
        # Calculate velocities
        dt = np.diff(times)
        velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
        mean_velocity = np.mean(np.sqrt(np.sum(velocities**2, axis=1)))
        
        # Calculate accelerations
        if len(positions) > 2:
            dv = np.diff(velocities, axis=0)
            dt2 = np.diff(times[1:])
            accelerations = dv / dt2[:, np.newaxis]
            mean_acceleration = np.mean(np.sqrt(np.sum(accelerations**2, axis=1)))
        else:
            mean_acceleration = 0
            
        return mean_velocity, mean_acceleration
    
    def calculate_confinement_ratio(self, positions: np.ndarray) -> float:
        """Calculate confinement ratio"""
        if len(positions) < 2:
            return 0
            
        # Calculate maximum distance from start
        distances_from_start = np.sqrt(np.sum((positions - positions[0])**2, axis=1))
        max_distance = np.max(distances_from_start)
        
        # Calculate total path length
        segment_lengths = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_length = np.sum(segment_lengths)
        
        # Calculate ratio
        if total_length > 0:
            confinement = max_distance / total_length
        else:
            confinement = 0
            
        return confinement