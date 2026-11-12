# src/particle_analysis/core/feature_calculation.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
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
            if not np.all(np.isfinite(positions)):
                return None
            
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
        """Fit MSD = 4 D t^alpha by a straight line in log-log space.

        log MSD = log(4 D) + alpha log t, so alpha is the slope and D = exp(intercept) / 4 (in the units of the
        positions and times, here pixels^2 per frame^alpha). Only lags with a positive MSD are used (log(0) is -inf;
        a track that returns exactly to an earlier position has MSD 0 at that lag).

        Returns (nan, nan) when the fit is invalid: fewer than two usable lags, or a negative slope (MSD falling with
        lag, which no diffusion model allows; it happens for immobile particles dominated by localisation noise).
        """
        msd_values = np.asarray(msd_values, dtype=float)
        time_lags = np.asarray(time_lags, dtype=float)
        keep = (msd_values > 0) & (time_lags > 0) & np.isfinite(msd_values) & np.isfinite(time_lags)
        if np.count_nonzero(keep) < 2:
            return np.nan, np.nan
        log_msd = np.log(msd_values[keep])
        log_time = np.log(time_lags[keep])
        
        # Perform linear fit
        slope, intercept, _, _, _ = stats.linregress(log_time, log_msd)
        if not np.isfinite(slope) or slope < 0:
            return np.nan, np.nan
        
        # Extract parameters
        alpha = slope
        diff_coef = np.exp(intercept) / 4
        
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
        
        # Eigenvalues of the symmetric covariance matrix (eigvalsh: real, ascending)
        eigenvals = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate asymmetry
        if len(eigenvals) > 1 and np.sum(eigenvals) > 0:
            asymmetry = 1 - (eigenvals[0] / eigenvals[-1])
        else:
            asymmetry = 0
            
        return float(asymmetry)
    
    def calculate_fractal_dimension(self, positions: np.ndarray) -> float:
        """Fractal dimension of the trajectory by Katz's method (Katz & George 1985).

        D = log(n) / (log(n) + log(d / L)), with n the number of steps, L the path length and d the planar extent
        (largest distance from the first point). D = 1 for a straight line, about 2 for a 2D random walk
        (1.9 on average for 15-1000 steps), above 2 for confined motion. NaN if there are fewer than 2 steps or
        the particle never moves.
        """
        positions = np.asarray(positions, dtype=float)
        n = len(positions) - 1
        if n < 2:
            return np.nan
        steps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        path_length = np.sum(steps)
        extent = np.max(np.sqrt(np.sum((positions - positions[0])**2, axis=1)))
        if path_length <= 0 or extent <= 0:
            return np.nan
        # denominator log(n d / L) <= 0 when the extent is at most one mean step (back-and-forth): undefined
        denominator = np.log(n) + np.log(extent / path_length)
        if denominator <= 0:
            return np.nan
        return float(np.log(n) / denominator)
    
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