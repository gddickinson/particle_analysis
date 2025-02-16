# src/particle_analysis/analysis/statistics.py

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging
from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures

# Set up logging
logger = logging.getLogger(__name__)

class TrackAnalyzer:
    """Class for analyzing track features and statistics"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.scaler = StandardScaler()
        
    def classify_tracks(self, 
                       features: List[TrackFeatures],
                       n_classes: int = 3) -> np.ndarray:
        """
        Classify tracks based on their features
        
        Parameters
        ----------
        features : List[TrackFeatures]
            List of track features
        n_classes : int
            Number of classes to identify
            
        Returns
        -------
        np.ndarray
            Array of class labels
        """
        try:
            # Extract feature matrix
            feature_matrix = self._create_feature_matrix(features)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_classes, random_state=42)
            labels = kmeans.fit_predict(scaled_features)
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in track classification: {str(e)}")
            return np.zeros(len(features))
            
    def compare_conditions(self,
                          condition1: List[TrackFeatures],
                          condition2: List[TrackFeatures],
                          feature_names: Optional[List[str]] = None
                          ) -> Dict[str, Dict]:
        """
        Compare features between two conditions
        
        Parameters
        ----------
        condition1 : List[TrackFeatures]
            Features from first condition
        condition2 : List[TrackFeatures]
            Features from second condition
        feature_names : List[str], optional
            Names of features to compare
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary of statistical tests for each feature
        """
        try:
            if feature_names is None:
                # Get all numeric features
                sample_dict = condition1[0].to_dict()
                feature_names = [
                    name for name, value in sample_dict.items()
                    if isinstance(value, (int, float)) and name != 'track_id'
                ]
                
            results = {}
            for feature in feature_names:
                values1 = [getattr(f, feature) for f in condition1]
                values2 = [getattr(f, feature) for f in condition2]
                
                # Perform statistical tests
                t_stat, t_pval = stats.ttest_ind(values1, values2)
                u_stat, u_pval = stats.mannwhitneyu(values1, values2,
                                                   alternative='two-sided')
                
                # Calculate effect size (Cohen's d)
                d = (np.mean(values1) - np.mean(values2)) / np.sqrt(
                    (np.var(values1) + np.var(values2)) / 2
                )
                
                results[feature] = {
                    'mean1': np.mean(values1),
                    'mean2': np.mean(values2),
                    'std1': np.std(values1),
                    'std2': np.std(values2),
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pval,
                    'cohens_d': d
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error in condition comparison: {str(e)}")
            return {}
            
    def analyze_mobility(self,
                        tracks: List[Track],
                        features: List[TrackFeatures],
                        time_window: int = 10) -> Dict[str, np.ndarray]:
        """
        Analyze track mobility patterns
        
        Parameters
        ----------
        tracks : List[Track]
            List of tracks to analyze
        features : List[TrackFeatures]
            Track features
        time_window : int
            Window size for mobility analysis
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of mobility metrics
        """
        try:
            results = {
                'instant_velocities': [],
                'turning_angles': [],
                'confinement_ratios': []
            }
            
            for track in tracks:
                positions = track.positions
                times = track.frames
                
                if len(positions) < time_window:
                    continue
                    
                # Calculate instantaneous velocities
                velocities = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                results['instant_velocities'].extend(velocities)
                
                # Calculate turning angles
                vectors = np.diff(positions, axis=0)
                angles = np.arctan2(vectors[1:, 1], vectors[1:, 0]) - \
                        np.arctan2(vectors[:-1, 1], vectors[:-1, 0])
                angles = np.rad2deg(angles)
                angles = (angles + 180) % 360 - 180  # Convert to [-180, 180]
                results['turning_angles'].extend(angles)
                
                # Calculate confinement ratios in windows
                for i in range(len(positions) - time_window):
                    window = positions[i:i+time_window]
                    total_path = np.sum(
                        np.sqrt(np.sum(np.diff(window, axis=0)**2, axis=1))
                    )
                    net_displacement = np.sqrt(
                        np.sum((window[-1] - window[0])**2)
                    )
                    ratio = net_displacement / total_path if total_path > 0 else 0
                    results['confinement_ratios'].append(ratio)
                    
            return {k: np.array(v) for k, v in results.items()}
            
        except Exception as e:
            logger.error(f"Error in mobility analysis: {str(e)}")
            return {}
            
    def calculate_spatial_statistics(self,
                                   tracks: List[Track],
                                   frame_size: Tuple[int, int],
                                   bin_size: int = 10
                                   ) -> Dict[str, np.ndarray]:
        """
        Calculate spatial statistics for tracks
        
        Parameters
        ----------
        tracks : List[Track]
            List of tracks to analyze
        frame_size : Tuple[int, int]
            Size of imaging frame
        bin_size : int
            Size of spatial bins
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of spatial statistics
        """
        try:
            # Create spatial grid
            n_bins_x = frame_size[0] // bin_size
            n_bins_y = frame_size[1] // bin_size
            density_map = np.zeros((n_bins_x, n_bins_y))
            velocity_map = np.zeros((n_bins_x, n_bins_y))
            
            for track in tracks:
                positions = track.positions
                
                # Calculate bin indices
                bin_x = np.clip(positions[:, 0] // bin_size, 0, n_bins_x-1)
                bin_y = np.clip(positions[:, 1] // bin_size, 0, n_bins_y-1)
                
                # Update density map
                for x, y in zip(bin_x, bin_y):
                    density_map[int(x), int(y)] += 1
                    
                # Calculate velocities
                if len(positions) > 1:
                    velocities = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                    for i, (x, y) in enumerate(zip(bin_x[:-1], bin_y[:-1])):
                        velocity_map[int(x), int(y)] += velocities[i]
                        
            # Normalize velocity map by density
            with np.errstate(divide='ignore', invalid='ignore'):
                velocity_map = np.divide(velocity_map, density_map,
                                      where=density_map!=0)
                
            return {
                'density_map': density_map,
                'velocity_map': velocity_map
            }
            
        except Exception as e:
            logger.error(f"Error in spatial statistics: {str(e)}")
            return {}
            
    def _create_feature_matrix(self, features: List[TrackFeatures]) -> np.ndarray:
        """Create feature matrix for classification"""
        # Get feature names
        sample_dict = features[0].to_dict()
        feature_names = [
            name for name, value in sample_dict.items()
            if isinstance(value, (int, float)) and name != 'track_id'
            and not isinstance(value, list)  # Exclude MSD values
        ]
        
        # Create matrix
        feature_matrix = np.zeros((len(features), len(feature_names)))
        for i, feature in enumerate(features):
            feature_matrix[i] = [getattr(feature, name) for name in feature_names]
            
        return feature_matrix
        
    def generate_report(self,
                       tracks: List[Track],
                       features: List[TrackFeatures],
                       output_path: str) -> bool:
        """
        Generate comprehensive analysis report
        
        Parameters
        ----------
        tracks : List[Track]
            List of tracks
        features : List[TrackFeatures]
            Track features
        output_path : str
            Path to save report
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            # Create summary statistics
            summary_stats = {
                'n_tracks': len(tracks),
                'n_particles': sum(len(t.particles) for t in tracks),
                'mean_track_length': np.mean([len(t.particles) for t in tracks]),
                'mean_diffusion_coef': np.mean([f.diffusion_coefficient 
                                              for f in features]),
                'mean_alpha': np.mean([f.alpha for f in features])
            }
            
            # Classify tracks
            labels = self.classify_tracks(features)
            track_classes = {
                'class_0': np.sum(labels == 0),
                'class_1': np.sum(labels == 1),
                'class_2': np.sum(labels == 2)
            }
            
            # Analyze mobility
            mobility = self.analyze_mobility(tracks, features)
            mobility_stats = {
                'mean_velocity': np.mean(mobility['instant_velocities']),
                'mean_turning_angle': np.mean(mobility['turning_angles']),
                'mean_confinement': np.mean(mobility['confinement_ratios'])
            }
            
            # Create report dataframes
            summary_df = pd.DataFrame([summary_stats])
            classes_df = pd.DataFrame([track_classes])
            mobility_df = pd.DataFrame([mobility_stats])
            
            # Save to Excel
            with pd.ExcelWriter(output_path) as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                classes_df.to_excel(writer, sheet_name='Classification', index=False)
                mobility_df.to_excel(writer, sheet_name='Mobility', index=False)
                
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return False