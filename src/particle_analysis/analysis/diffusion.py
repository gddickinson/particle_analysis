# src/particle_analysis/analysis/diffusion.py

import numpy as np
from scipy import stats
from functools import lru_cache
from scipy.optimize import least_squares
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
    UNCLASSIFIED = 'unclassified'  # the MSD curve is too short or the fits failed

@lru_cache(maxsize=64)
def msd_covariance(n_positions: int, lags: Tuple[int, ...]) -> np.ndarray:
    """Covariance (up to a scale factor) of the time-averaged MSD at the given lags for a 2D Brownian track of
    n_positions points with unit step variance per axis (localisation noise ignored).

    For Gaussian displacements, Cov(r_a^2, r_b^2) = 4 s^4 o^2 in 2D, with o the number of steps the two
    displacement windows share; Cov(MSD_n, MSD_m) averages this over all window pairs (cf. Qian et al. 1991).
    """
    N = n_positions
    C = np.zeros((len(lags), len(lags)))
    for a, n in enumerate(lags):
        kn = N - n
        for b in range(a, len(lags)):
            m = lags[b]
            km = N - m
            d = np.arange(-(kn - 1), km)  # offset j - i between the windows' starts
            count = np.clip(np.minimum(kn - np.maximum(0, -d), km - np.maximum(0, d)), 0, None)
            overlap = np.clip(np.minimum(n, d + m) - np.maximum(0, d), 0, None).astype(float)
            C[a, b] = C[b, a] = 4 * np.sum(count * overlap**2) / (kn * km)
    return C


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
    
    # number of free parameters of each MSD model (for the information criterion)
    N_PARAMS = {'normal': 1, 'anomalous': 2, 'confined': 2, 'directed': 2}
    MIN_MSD_POINTS = 4  # the criterion of a 2-parameter model needs n > k + 1

    def fit_diffusion_models(self,
                            time_lags: np.ndarray,
                            msd_values: np.ndarray,
                            n_positions: Optional[int] = None
                            ) -> Dict[str, Dict]:
        """
        Fit different diffusion models to MSD curve
        
        Each model is fitted separately (a failed fit leaves the others). MSD values at different lags of one
        track are strongly correlated; when ``n_positions`` (the track length) is given, the fits are generalised
        least squares with the covariance of the MSD estimator for 2D Brownian motion (``msd_covariance``), so
        the residual chi^2 is meaningful for model selection. Without it the fits are ordinary least squares.
        
        Every result has 'fit_values', 'r_squared' (unweighted), 'chi2' (weighted residual sum of squares),
        'n_params' and 'bic' = n ln(chi2/n) + k ln(n), the Bayesian information criterion, which penalises
        the extra parameter of the anomalous, confined and directed models (NaN when n <= k + 1).
        
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
        time_lags = np.asarray(time_lags, dtype=float)
        msd_values = np.asarray(msd_values, dtype=float)
        results = {}
        if len(msd_values) == 0 or not np.all(np.isfinite(msd_values)):
            return results

        # Normal diffusion: MSD = 4Dt
        def normal_diffusion(t, D):
            return 4 * D * t

        # Anomalous diffusion: MSD = 4Dt^alpha
        def anomalous_diffusion(t, D, alpha):
            return 4 * D * np.power(t, alpha)

        # Confined diffusion: MSD = A(1 - exp(-4Dt/A))
        def confined_diffusion(t, D, A):
            return A * (1 - np.exp(-4 * D * t / A))

        # Directed motion: MSD = 4Dt + (vt)^2
        def directed_diffusion(t, D, v):
            return 4 * D * t + (v * t)**2

        msd_max = max(float(np.max(msd_values)), 1e-12)
        specs = {
            'normal': (normal_diffusion, [0.1], (0, np.inf), ['D']),
            'anomalous': (anomalous_diffusion, [0.1, 1.0], ([0, 0], [np.inf, 2]), ['D', 'alpha']),
            'confined': (confined_diffusion, [0.1, msd_max], (0, np.inf), ['D', 'A']),
            'directed': (directed_diffusion, [0.1, 1.0], (0, np.inf), ['D', 'velocity']),
        }
        # whitening matrix W with W^T W = C^-1 (identity for ordinary least squares)
        W = np.eye(len(msd_values))
        if n_positions is not None:
            lags = np.rint(time_lags / self.frame_interval).astype(int)
            try:
                W = np.linalg.inv(np.linalg.cholesky(msd_covariance(int(n_positions), tuple(lags))))
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"MSD covariance not usable, fitting unweighted: {str(e)}")
        for name, (func, p0, bounds, names) in specs.items():
            try:
                fit = least_squares(lambda p: W @ (msd_values - func(time_lags, *p)), p0,
                                    bounds=bounds if isinstance(bounds[0], list) else
                                    ([bounds[0]] * len(p0), [bounds[1]] * len(p0)))
                if not fit.success:
                    raise RuntimeError(fit.message)
                popt = fit.x
            except Exception as e:
                logger.warning(f"{name} diffusion fit failed: {str(e)}")
                continue
            fit_values = func(time_lags, *popt)
            res = dict(zip(names, popt))
            if name == 'confined':
                res['confinement_size'] = np.sqrt(res.pop('A'))
            res['fit_values'] = fit_values
            res['r_squared'] = self._r_squared(msd_values, fit_values)
            res['chi2'] = float(np.sum((W @ (msd_values - fit_values))**2))
            res['n_params'] = self.N_PARAMS[name]
            res['bic'] = self._bic(res['chi2'], len(msd_values), self.N_PARAMS[name])
            results[name] = res

        return results

    @staticmethod
    def _bic(chi2: float, n: int, k: int) -> float:
        """BIC for a (weighted) least-squares fit with unknown noise scale: n ln(chi2/n) + k ln(n)."""
        if n <= k + 1:
            return np.nan
        return n * np.log(max(chi2, 1e-300) / n) + k * np.log(n)

    def select_model(self, fit_results: Dict[str, Dict]) -> Optional[str]:
        """Model with the lowest BIC among the fitted ones, or None if none can be compared."""
        scores = {k: v['bic'] for k, v in fit_results.items() if np.isfinite(v.get('bic', np.nan))}
        if not scores:
            return None
        return min(scores.items(), key=lambda x: x[1])[0]
    
    def classify_motion(self,
                       track: Track,
                       max_points: Optional[int] = None) -> MotionType:
        """
        Classify type of motion based on MSD analysis
        
        The MSD curve is fitted with the four models (generalised least squares, see fit_diffusion_models) and
        the one with the lowest BIC (which penalises the extra parameter of the anomalous, confined and directed
        models) is chosen. If that is the anomalous
        model, alpha < 0.8 is called confined and alpha > 1.2 directed. Tracks with fewer than
        MIN_MSD_POINTS MSD points, or whose fits all fail, are UNCLASSIFIED.
        
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
            time_lags, msd_values = self.calculate_msd(track, max_points)
            if len(msd_values) < self.MIN_MSD_POINTS:
                return MotionType.UNCLASSIFIED
            fit_results = self.fit_diffusion_models(time_lags, msd_values, len(track.particles))
            return self._motion_from_fits(fit_results)
        except Exception as e:
            logger.error(f"Error classifying motion: {str(e)}")
            return MotionType.UNCLASSIFIED

    def _motion_from_fits(self, fit_results: Dict[str, Dict]) -> MotionType:
        best_model = self.select_model(fit_results)
        if best_model is None:
            return MotionType.UNCLASSIFIED
        if best_model == 'anomalous':
            alpha = fit_results['anomalous']['alpha']
            if alpha < 0.8:
                return MotionType.CONFINED
            elif alpha > 1.2:
                return MotionType.DIRECTED
            return MotionType.NORMAL_DIFFUSION
        return {'confined': MotionType.CONFINED, 'directed': MotionType.DIRECTED,
                'normal': MotionType.NORMAL_DIFFUSION}[best_model]
    
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
            
            # Fit models and classify (same rule as classify_motion, without refitting)
            fit_results = self.fit_diffusion_models(time_lags, msd_values, len(track.particles))
            if len(msd_values) < self.MIN_MSD_POINTS:
                motion_type = MotionType.UNCLASSIFIED
            else:
                motion_type = self._motion_from_fits(fit_results)
            
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