# src/particle_analysis/visualization/plot_utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import pyqtgraph as pg
from typing import List, Dict, Optional, Tuple, Union
import logging
from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures

# Set up logging
logger = logging.getLogger(__name__)

class TrackVisualizer:
    """Class for creating visualizations of particle tracks and analysis results"""
    
    def __init__(self, 
                 colormap: str = 'viridis',
                 dpi: int = 300,
                 style: str = 'default'):
        """
        Initialize visualizer
        
        Parameters
        ----------
        colormap : str
            Matplotlib colormap name
        dpi : int
            DPI for saved figures
        style : str
            Matplotlib style
        """
        self.colormap = colormap
        self.dpi = dpi
        plt.style.use(style)
        
    def plot_tracks(self,
                   tracks: List[Track],
                   features: Optional[List[TrackFeatures]] = None,
                   color_by: str = 'track_id',
                   ax: Optional[plt.Axes] = None,
                   show_points: bool = True,
                   min_alpha: float = 0.3,
                   title: Optional[str] = None) -> plt.Axes:
        """
        Plot particle tracks
        
        Parameters
        ----------
        tracks : List[Track]
            List of tracks to plot
        features : List[TrackFeatures], optional
            Track features for coloring
        color_by : str
            Property to color tracks by ('track_id', 'intensity', 'diffusion', etc.)
        ax : plt.Axes, optional
            Axes to plot on
        show_points : bool
            Whether to show individual points
        min_alpha : float
            Minimum alpha value for tracks
        title : str, optional
            Plot title
            
        Returns
        -------
        plt.Axes
            Matplotlib axes with plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
            
        # Create color mapping
        if color_by == 'track_id':
            colors = plt.cm.get_cmap(self.colormap)(
                np.linspace(0, 1, len(tracks))
            )
        elif features is not None and hasattr(features[0], color_by):
            values = [getattr(f, color_by) for f in features]
            norm = plt.Normalize(min(values), max(values))
            colors = plt.cm.get_cmap(self.colormap)(norm(values))
        else:
            colors = [plt.cm.get_cmap(self.colormap)(0.5)] * len(tracks)
            
        # Plot each track
        for track, color in zip(tracks, colors):
            positions = track.positions
            times = track.frames
            
            # Calculate alpha based on time
            if len(times) > 1:
                alphas = np.linspace(min_alpha, 1, len(times))
            else:
                alphas = [1]
                
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1],
                   color=color, alpha=0.7, linewidth=1)
            
            # Plot points if requested
            if show_points:
                scatter = ax.scatter(positions[:, 0], positions[:, 1],
                                   c=alphas, cmap='Greys',
                                   s=20, color=color)
                
        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        return ax
    
    def plot_track_feature_distributions(self,
                                       features: List[TrackFeatures],
                                       feature_names: Optional[List[str]] = None,
                                       fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Plot distributions of track features
        
        Parameters
        ----------
        features : List[TrackFeatures]
            List of track features
        feature_names : List[str], optional
            Names of features to plot (default: all numeric features)
        fig : plt.Figure, optional
            Figure to plot on
            
        Returns
        -------
        plt.Figure
            Matplotlib figure with plots
        """
        if feature_names is None:
            # Get all numeric features
            sample_dict = features[0].to_dict()
            feature_names = [
                name for name, value in sample_dict.items()
                if isinstance(value, (int, float)) and name != 'track_id'
            ]
            
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if fig is None:
            fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
            
        for i, feature_name in enumerate(feature_names, 1):
            ax = fig.add_subplot(n_rows, n_cols, i)
            values = [getattr(f, feature_name) for f in features]
            
            # Create histogram
            ax.hist(values, bins='auto', density=True, alpha=0.7)
            ax.set_xlabel(feature_name.replace('_', ' ').title())
            ax.set_ylabel('Density')
            
        fig.tight_layout()
        return fig
    
    def plot_msd_curves(self,
                       features: List[TrackFeatures],
                       ax: Optional[plt.Axes] = None,
                       color_by: str = 'diffusion_coefficient',
                       show_fits: bool = True) -> plt.Axes:
        """
        Plot MSD curves for tracks
        
        Parameters
        ----------
        features : List[TrackFeatures]
            List of track features
        ax : plt.Axes, optional
            Axes to plot on
        color_by : str
            Property to color curves by
        show_fits : bool
            Whether to show power law fits
            
        Returns
        -------
        plt.Axes
            Matplotlib axes with plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Create color mapping
        values = [getattr(f, color_by) for f in features]
        norm = plt.Normalize(min(values), max(values))
        cmap = plt.cm.get_cmap(self.colormap)
        
        # Plot each MSD curve
        for feature, value in zip(features, values):
            color = cmap(norm(value))
            time_lags = np.arange(len(feature.msd_values))
            
            # Plot MSD curve
            ax.plot(time_lags, feature.msd_values,
                   color=color, alpha=0.5)
            
            # Plot power law fit if requested
            if show_fits:
                fit = feature.diffusion_coefficient * (4 * time_lags**feature.alpha)
                ax.plot(time_lags, fit,
                       color=color, alpha=0.8, linestyle='--')
                
        ax.set_xlabel('Time Lag')
        ax.set_ylabel('MSD')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, label=color_by.replace('_', ' ').title())
        
        return ax
    
    def create_track_movie(self,
                          tracks: List[Track],
                          output_path: str,
                          frame_range: Optional[Tuple[int, int]] = None,
                          fps: int = 10,
                          tail_length: int = 10) -> bool:
        """
        Create movie of particle tracks
        
        Parameters
        ----------
        tracks : List[Track]
            List of tracks to animate
        output_path : str
            Path to save movie
        frame_range : Tuple[int, int], optional
            Range of frames to include
        fps : int
            Frames per second
        tail_length : int
            Length of track history to show
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            # Get frame range
            if frame_range is None:
                start_frame = min(t.start_frame for t in tracks)
                end_frame = max(t.end_frame for t in tracks)
                frame_range = (start_frame, end_frame)
                
            fig, ax = plt.subplots(figsize=(8, 8))
            frames = []
            
            # Create color mapping
            colors = plt.cm.get_cmap(self.colormap)(
                np.linspace(0, 1, len(tracks))
            )
            
            # Process each frame
            for frame in range(frame_range[0], frame_range[1] + 1):
                ax.clear()
                
                # Plot each track
                for track, color in zip(tracks, colors):
                    # Get positions up to current frame
                    mask = track.frames <= frame
                    positions = track.positions[mask]
                    
                    if len(positions) > 0:
                        # Plot recent history
                        if tail_length > 0:
                            start_idx = max(0, len(positions) - tail_length)
                            history = positions[start_idx:]
                            ax.plot(history[:, 0], history[:, 1],
                                  color=color, alpha=0.5)
                            
                        # Plot current position
                        ax.scatter(positions[-1, 0], positions[-1, 1],
                                 color=color, s=50)
                        
                ax.set_aspect('equal')
                ax.set_title(f'Frame {frame}')
                
                # Save frame
                frames.append(self._get_frame_image(fig))
                
            # Save movie
            import imageio
            imageio.mimsave(output_path, frames, fps=fps)
            plt.close(fig)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating track movie: {str(e)}")
            return False
            
    @staticmethod
    def _get_frame_image(fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to image array"""
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image