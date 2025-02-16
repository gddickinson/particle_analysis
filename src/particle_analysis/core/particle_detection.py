# src/particle_analysis/core/particle_detection.py

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Particle:
    """Data class for storing particle information"""
    frame: int
    x: float
    y: float
    intensity: float
    sigma: float
    snr: float
    frame_size: Tuple[int, int]
    id: Optional[int] = None

    def to_dict(self):
        """Convert particle data to dictionary"""
        return {
            'frame': self.frame,
            'x': self.x,
            'y': self.y,
            'intensity': self.intensity,
            'sigma': self.sigma,
            'snr': self.snr,
            'id': self.id
        }

class ParticleDetector:
    """Class for detecting particles in fluorescence microscopy images"""

    def __init__(self,
                 min_sigma: float = 1.0,
                 max_sigma: float = 3.0,
                 sigma_steps: int = 5,
                 threshold_rel: float = 0.2,
                 min_distance: int = 5,
                 exclude_border: int = True):
        """
        Initialize particle detector with detection parameters

        Parameters
        ----------
        min_sigma : float
            Minimum standard deviation for Gaussian kernel
        max_sigma : float
            Maximum standard deviation for Gaussian kernel
        sigma_steps : int
            Number of steps between min and max sigma
        threshold_rel : float
            Relative threshold for peak detection
        min_distance : int
            Minimum distance between peaks
        exclude_border : bool
            Whether to exclude peaks near image border
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_steps = sigma_steps
        self.threshold_rel = threshold_rel
        self.min_distance = min_distance
        self.exclude_border = exclude_border

        # Calculate sigma values
        self.sigmas = np.linspace(min_sigma, max_sigma, sigma_steps)

    def detect_frame(self, frame: np.ndarray) -> List[Particle]:
        """
        Detect particles in a single frame

        Parameters
        ----------
        frame : np.ndarray
            2D image array

        Returns
        -------
        List[Particle]
            List of detected particles
        """
        try:
            # Validate input
            if frame.ndim != 2:
                raise ValueError("Input frame must be 2D array")

            particles = []
            frame_float = frame.astype(float)

            # Calculate background and noise
            background = ndimage.median_filter(frame_float, size=20)
            noise = np.std(frame_float - background)

            # Subtract background
            frame_bg_sub = frame_float - background

            # Find local maxima
            coords = peak_local_max(
                frame_bg_sub,
                min_distance=self.min_distance,
                threshold_rel=self.threshold_rel,
                exclude_border=self.exclude_border
            )

            # Fit Gaussian to each peak
            for coord in coords:
                y, x = coord
                try:
                    # Extract region around peak
                    size = int(4 * self.max_sigma)
                    y_min = max(0, y - size)
                    y_max = min(frame.shape[0], y + size + 1)
                    x_min = max(0, x - size)
                    x_max = min(frame.shape[1], x + size + 1)
                    region = frame_bg_sub[y_min:y_max, x_min:x_max]

                    # Fit 2D Gaussian
                    params = self._fit_gaussian(region)
                    if params is not None:
                        amplitude, x_fit, y_fit, sigma = params

                        # Calculate absolute positions
                        x_abs = x_min + x_fit
                        y_abs = y_min + y_fit

                        # Calculate SNR
                        snr = amplitude / noise

                        # Create particle object
                        particle = Particle(
                            frame=0,  # Frame number added later
                            x=x_abs,
                            y=y_abs,
                            intensity=amplitude,
                            sigma=sigma,
                            snr=snr,
                            frame_size=frame.shape
                        )
                        particles.append(particle)

                except Exception as e:
                    logger.warning(f"Failed to fit particle at ({x}, {y}): {str(e)}")
                    continue

            return particles

        except Exception as e:
            logger.error(f"Error in particle detection: {str(e)}")
            raise

    def detect_movie(self, movie: np.ndarray) -> List[Particle]:
        """
        Detect particles in all frames of a movie

        Parameters
        ----------
        movie : np.ndarray
            3D array of frames

        Returns
        -------
        List[Particle]
            List of detected particles across all frames
        """
        if movie.ndim != 3:
            raise ValueError("Input movie must be 3D array")

        all_particles = []
        for frame_num in range(movie.shape[0]):
            frame_particles = self.detect_frame(movie[frame_num])
            # Add frame number to particles
            for particle in frame_particles:
                particle.frame = frame_num
            all_particles.extend(frame_particles)

        return all_particles

    def _fit_gaussian(self, region: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Fit 2D Gaussian to image region

        Parameters
        ----------
        region : np.ndarray
            2D array containing region around peak

        Returns
        -------
        Tuple[float, float, float, float]
            (amplitude, x_center, y_center, sigma) or None if fit fails
        """
        try:
            # Ensure region is float type
            region = region.astype(float)

            # Get initial estimates
            background = np.percentile(region, 10)
            region_bg_sub = region - background
            amplitude = np.max(region_bg_sub)

            # Find center of mass for better initial position estimate
            total_intensity = np.sum(region_bg_sub)
            if total_intensity <= 0:
                return None

            y, x = np.indices(region.shape)
            x_center_init = np.sum(x * region_bg_sub) / total_intensity
            y_center_init = np.sum(y * region_bg_sub) / total_intensity

            # Estimate initial sigma using intensity-weighted variance
            x_var = np.sum((x - x_center_init)**2 * region_bg_sub) / total_intensity
            y_var = np.sum((y - y_center_init)**2 * region_bg_sub) / total_intensity
            sigma_init = np.sqrt((x_var + y_var) / 2)

            # Constrain sigma to reasonable range
            sigma_init = np.clip(sigma_init, 1.0, 3.0)

            # Define 2D Gaussian function
            def gaussian_2d(coords, amplitude, x_center, y_center, sigma):
                y, x = coords
                gauss = amplitude * np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))
                return gauss.ravel()

            # Prepare data for fitting
            coords = np.array([y, x])
            data = region_bg_sub

            # Set bounds based on region size
            height, width = region.shape
            bounds = (
                [amplitude * 0.5, 0, 0, 0.8],  # lower bounds
                [amplitude * 2.0, width, height, 4.0]  # upper bounds
            )

            # Perform fit with increased max iterations and tolerance
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(
                gaussian_2d,
                coords,
                data.ravel(),
                p0=[amplitude, x_center_init, y_center_init, sigma_init],
                bounds=bounds,
                maxfev=1000,  # Increase max function evaluations
                ftol=1e-4,    # Relax fitting tolerance
                xtol=1e-4
            )

            # Check fit quality
            fitted_gauss = gaussian_2d(coords, *popt).reshape(region.shape)
            residuals = data - fitted_gauss
            rms_error = np.sqrt(np.mean(residuals**2))
            rel_error = rms_error / amplitude

            # Reject poor fits
            if rel_error > 0.5:  # 50% relative error threshold
                return None

            # Check if fitted parameters are reasonable
            if not (0.5 <= popt[3] <= 4.0):  # sigma bounds
                return None

            return tuple(popt)

        except Exception as e:
            logger.warning(f"Gaussian fitting failed: {str(e)}")
            return None

    @staticmethod
    def create_gaussian_kernel(sigma: float, size: int) -> np.ndarray:
        """
        Create 2D Gaussian kernel

        Parameters
        ----------
        sigma : float
            Standard deviation of Gaussian
        size : int
            Size of kernel (must be odd)

        Returns
        -------
        np.ndarray
            2D Gaussian kernel
        """
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        x = np.arange(size) - (size - 1) / 2
        gauss = np.exp(-x**2 / (2 * sigma**2))
        kernel = np.outer(gauss, gauss)
        return kernel / kernel.sum()
