# src/particle_analysis/analysis/diffusion.py

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional
import logging
from enum import Enum
from ..core.particle_tracking import Track

# Set up logging
logger = logging.getLogger(__name__)

class MotionType(Enum):
    """Enumeration of different motion types"""
    CONFINED = 'confined'
    NORMAL_DIFFUSION = 'normal_diffusion'
    DIRECTED = 'directed'
    ANOMALOUS = 'anomalous'

class DiffusionAnalyzer:
    """Class for analyzing particle diffusion and motion types"""
    
    def __init__(self, 
                 pixel_size: float = 0.108,  # microns per pixel
                 frame_interval: float = 0.1):  # seconds
        """
        Initialize analyzer
        
        Parameters
        ----------
        pixel_size : float
            Size of pixel in microns
        frame_interval : float
            Time between frames in seconds
        """
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        
    def calculate_msd(self, 
                     track: Track,
                     max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Mean Square Displacement curve
        
        Parameters
        ----------
        track : Track
            Track to analyze
        max_points : int, optional
            Maximum number of points to calculate
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time lags and MSD values
        """
        positions = track.positions * self.pixel_size  # Convert to microns
        n_points = len(positions)
        
        if max_points is None:
            max_points = n_points // 4  # Use up to 1/4 of track length
        else:
            max_points = min(max_points, n_points - 1)
            
        # Calculate MSD for each time lag
        time_lags = np.arange(1, max_points + 1) * self.frame_interval
        msd_values = np.zeros(max_points)
        
        for i in range(max_points):
            lag = i + 1
            # Calculate displacements for this lag
            disp = positions[lag:] - positions[:-lag]
            # Calculate squared displacements
            squared_disp = np.sum(disp**2, axis=1)
            # Average over all pairs
            msd_values[i] = np.mean(squared_disp)
            
        return time_lags, msd_values
    
    def fit_diffusion_models(self,
                            time_lags: np.ndarray,
                            msd_values: np.ndarray
                            ) -> Dict[str, Dict]:
        """
        Fit different diffusion models to MSD curve
        
        Parameters
        ----------
        time_lags : np.ndarray
            Time lag values
        msd_values : np.ndarray
            MSD values
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary of fit results for each model
        """
        results = {}
        
        try:
            # Normal diffusion: MSD = 4Dt
            def normal_diffusion(t, D):
                return 4 * D * t
            
            popt_normal, pcov_normal = curve_fit(
                normal_diffusion, time_lags, msd_values,
                p0=[0.1], bounds=(0, np.inf)
            )
            results['normal'] = {
                'D': popt_normal[0],
                'fit_values': normal_diffusion(time_lags, *popt_normal),
                'r_squared': self._r_squared(msd_values, 
                                          normal_diffusion(time_lags, *popt_normal))
            }
            
            # Anomalous diffusion: MSD = 4Dt^α
            def anomalous_diffusion(t, D, alpha):
                return 4 * D * np.power(t, alpha)
            
            popt_anom, pcov_anom = curve_fit(
                anomalous_diffusion, time_lags, msd_values,
                p0=[0.1, 1.0], bounds=([0, 0], [np.inf, 2])
            )
            results['anomalous'] = {
                'D': popt_anom[0],
                'alpha': popt_anom[1],
                'fit_values': anomalous_diffusion(time_lags, *popt_anom),
                'r_squared': self._r_squared(msd_values,
                                          anomalous_diffusion(time_lags, *popt_anom))
            }
            
            # Confined diffusion: MSD = A(1 - exp(-4Dt/A))
            def confined_diffusion(t, D, A):
                return A * (1 - np.exp(-4 * D * t / A))
            
            popt_conf, pcov_conf = curve_fit(
                confined_diffusion, time_lags, msd_values,
                p0=[0.1, np.max(msd_values)], bounds=(0, np.inf)
            )
            results['confined'] = {
                'D': popt_conf[0],
                'confinement_size': np.sqrt(popt_conf[1]),
                'fit_values': confined_diffusion(time_lags, *popt_conf),
                'r_squared': self._r_squared(msd_values,
                                          confined_diffusion(time_lags, *popt_conf))
            }
            
            # Directed motion: MSD = 4Dt + (vt)^2
            def directed_diffusion(t, D, v):
                return 4 * D * t + (v * t)**2
            
            popt_dir, pcov_dir = curve_fit(
                directed_diffusion, time_lags, msd_values,
                p0=[0.1, 1.0], bounds=(0, np.inf)
            )
            results['directed'] = {
                'D': popt_dir[0],
                'velocity': popt_dir[1],
                'fit_values': directed_diffusion(time_lags, *popt_dir),
                'r_squared': self._r_squared(msd_values,
                                          directed_diffusion(time_lags, *popt_dir))
            }
            
        except Exception as e:
            logger.error(f"Error fitting diffusion models: {str(e)}")
            
        return results
    
    def classify_motion(self,
                       track: Track,
                       max_points: Optional[int] = None) -> MotionType:
        """
        Classify type of motion based on MSD analysis
        
        Parameters
        ----------
        track : Track
            Track to analyze
        max_points : int, optional
            Maximum number of points for MSD calculation
            
        Returns
        -------
        MotionType
            Classified motion type
        """
        try:
            # Calculate MSD
            time_lags, msd_values = self.calculate_msd(track, max_points)
            
            # Fit models
            fit_results = self.fit_diffusion_models(time_lags, msd_values)
            
            # Compare fits using R-squared values
            r_squared_values = {k: v['r_squared'] for k, v in fit_results.items()}
            best_model = max(r_squared_values.items(), key=lambda x: x[1])[0]
            
            # Additional criteria for classification
            if best_model == 'anomalous':
                alpha = fit_results['anomalous']['alpha']
                if alpha < 0.8:
                    return MotionType.CONFINED
                elif alpha > 1.2:
                    return MotionType.DIRECTED
                else:
                    return MotionType.NORMAL_DIFFUSION
            elif best_model == 'confined':
                return MotionType.CONFINED
            elif best_model == 'directed':
                return MotionType.DIRECTED
            else:
                return MotionType.NORMAL_DIFFUSION
                
        except Exception as e:
            logger.error(f"Error classifying motion: {str(e)}")
            return MotionType.NORMAL_DIFFUSION
    
    def analyze_track(self, 
                     track: Track,
                     max_points: Optional[int] = None) -> Dict:
        """
        Perform comprehensive diffusion analysis on track
        
        Parameters
        ----------
        track : Track
            Track to analyze
        max_points : int, optional
            Maximum number of points for MSD calculation
            
        Returns
        -------
        Dict
            Dictionary of analysis results
        """
        try:
            # Calculate MSD
            time_lags, msd_values = self.calculate_msd(track, max_points)
            
            # Fit models
            fit_results = self.fit_diffusion_models(time_lags, msd_values)
            
            # Classify motion
            motion_type = self.classify_motion(track, max_points)
            
            # Calculate instantaneous diffusion coefficients
            inst_diff_coef = self.calculate_instantaneous_diffusion(track)
            
            return {
                'track_id': track.id,
                'motion_type': motion_type,
                'msd_data': {
                    'time_lags': time_lags,
                    'msd_values': msd_values
                },
                'fit_results': fit_results,
                'instantaneous_diffusion': inst_diff_coef
            }
            
        except Exception as e:
            logger.error(f"Error in track analysis: {str(e)}")
            return {}
    
    def calculate_instantaneous_diffusion(self, track: Track) -> np.ndarray:
        """
        Calculate instantaneous diffusion coefficients
        
        Parameters
        ----------
        track : Track
            Track to analyze
            
        Returns
        -------
        np.ndarray
            Array of instantaneous diffusion coefficients
        """
        positions = track.positions * self.pixel_size  # Convert to microns
        
        # Calculate instantaneous squared displacements
        displacements = np.diff(positions, axis=0)
        squared_displacements = np.sum(displacements**2, axis=1)
        
        # Calculate instantaneous diffusion coefficients
        inst_diff_coef = squared_displacements / (4 * self.frame_interval)
        
        return inst_diff_coef
    
    @staticmethod
    def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value for fit"""
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)