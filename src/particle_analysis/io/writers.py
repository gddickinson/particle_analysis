# src/particle_analysis/io/writers.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional
import logging
from ..core.particle_detection import Particle
from ..core.particle_tracking import Track
from ..core.feature_calculation import TrackFeatures

# Set up logging
logger = logging.getLogger(__name__)

class DataWriter:
    """Class for writing various data formats used in particle analysis"""

    def __init__(self):
        """Initialize writer"""
        pass

    def write_particles_csv(self,
                           particles: List[Particle],
                           file_path: Union[str, Path],
                           include_extra: bool = True) -> bool:
        """
        Write particle detections to CSV file

        Parameters
        ----------
        particles : List[Particle]
            List of detected particles
        file_path : str or Path
            Path to output CSV file
        include_extra : bool
            Whether to include extra particle properties

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Convert particles to dataframe
            data = []
            for p in particles:
                particle_dict = {
                    'frame': p.frame,
                    'x': p.x,
                    'y': p.y,
                    'intensity': p.intensity
                }

                if include_extra:
                    particle_dict.update({
                        'sigma': p.sigma,
                        'snr': p.snr,
                        'id': p.id if p.id is not None else -1
                    })

                data.append(particle_dict)

            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Saved particles to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing particles file {file_path}: {str(e)}")
            return False

    def write_tracks_csv(self,
                        tracks: List[Track],
                        file_path: Union[str, Path],
                        include_extra: bool = True) -> bool:
        """
        Write particle tracks to CSV file

        Parameters
        ----------
        tracks : List[Track]
            List of particle tracks
        file_path : str or Path
            Path to output CSV file
        include_extra : bool
            Whether to include extra track properties

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Convert tracks to dataframe
            data = []
            for track in tracks:
                for particle in track.particles:
                    particle_dict = {
                        'track_number': track.id,
                        'frame': particle.frame,
                        'x': particle.x,
                        'y': particle.y,
                        'intensity': particle.intensity
                    }

                    if include_extra:
                        particle_dict.update({
                            'sigma': particle.sigma,
                            'snr': particle.snr,
                            'id': particle.id if particle.id is not None else -1,
                            'start_frame': track.start_frame,
                            'end_frame': track.end_frame
                        })

                    data.append(particle_dict)

            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Saved tracks to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing tracks file {file_path}: {str(e)}")
            return False

    def write_features_csv(self,
                          features: List[TrackFeatures],
                          file_path: Union[str, Path]) -> bool:
        """
        Write track features to CSV file

        Parameters
        ----------
        features : List[TrackFeatures]
            List of track features
        file_path : str or Path
            Path to output CSV file

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Convert features to dataframe
            data = []
            for feature in features:
                feature_dict = feature.to_dict()
                # Convert numpy arrays to lists for JSON serialization
                feature_dict['msd_values'] = feature_dict['msd_values']
                data.append(feature_dict)

            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Saved features to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing features file {file_path}: {str(e)}")
            return False

    def write_analysis_summary(self,
                             tracks: List[Track],
                             features: List[TrackFeatures],
                             file_path: Union[str, Path]) -> bool:
        """
        Write comprehensive analysis summary to Excel file

        Parameters
        ----------
        tracks : List[Track]
            List of particle tracks
        features : List[TrackFeatures]
            List of track features
        file_path : str or Path
            Path to output Excel file

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Create Excel writer
            with pd.ExcelWriter(file_path) as writer:
                # Write track data
                track_data = []
                for track in tracks:
                    track_dict = {
                        'track_id': track.id,
                        'n_particles': len(track.particles),
                        'start_frame': track.start_frame,
                        'end_frame': track.end_frame,
                        'duration': track.end_frame - track.start_frame + 1,
                        'mean_intensity': np.mean([p.intensity for p in track.particles])
                    }
                    track_data.append(track_dict)

                track_df = pd.DataFrame(track_data)
                track_df.to_excel(writer, sheet_name='Tracks', index=False)

                # Write feature data
                feature_data = []
                for feature in features:
                    feature_dict = feature.to_dict()
                    feature_dict['msd_values'] = str(feature_dict['msd_values'])
                    feature_data.append(feature_dict)

                feature_df = pd.DataFrame(feature_data)
                feature_df.to_excel(writer, sheet_name='Features', index=False)

                # Write summary statistics
                summary_dict = {
                    'n_tracks': len(tracks),
                    'mean_track_length': np.mean([len(t.particles) for t in tracks]),
                    'mean_diffusion_coef': np.mean([f.diffusion_coefficient for f in features]),
                    'mean_alpha': np.mean([f.alpha for f in features]),
                    'mean_radius_gyration': np.mean([f.radius_gyration for f in features]),
                    'mean_straightness': np.mean([f.straightness for f in features]),
                    'mean_velocity': np.mean([f.mean_velocity for f in features]),
                    'std_diffusion_coef': np.std([f.diffusion_coefficient for f in features]),
                    'std_alpha': np.std([f.alpha for f in features]),
                    'std_radius_gyration': np.std([f.radius_gyration for f in features]),
                    'std_straightness': np.std([f.straightness for f in features]),
                    'std_velocity': np.std([f.mean_velocity for f in features])
                }

                summary_df = pd.DataFrame([summary_dict])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                logger.info(f"Saved analysis summary to {file_path}")
                return True

        except Exception as e:
            logger.error(f"Error writing analysis summary {file_path}: {str(e)}")
            return False

    def write_batch_results(self,
                           results: Dict[str, Dict],
                           file_path: Union[str, Path]) -> bool:
        """
        Write batch processing results to Excel file

        Parameters
        ----------
        results : Dict[str, Dict]
            Dictionary of results for each processed file
        file_path : str or Path
            Path to output Excel file

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Create Excel writer
            with pd.ExcelWriter(file_path) as writer:
                # Write summary for all files
                summary_data = []
                for filename, result in results.items():
                    file_summary = {
                        'filename': filename,
                        'n_tracks': result.get('n_tracks', 0),
                        'n_particles': result.get('n_particles', 0),
                        'mean_track_length': result.get('mean_track_length', 0),
                        'mean_diffusion_coef': result.get('mean_diffusion_coef', 0),
                        'mean_alpha': result.get('mean_alpha', 0),
                        'processing_time': result.get('processing_time', 0),
                        'status': result.get('status', 'unknown')
                    }
                    summary_data.append(file_summary)

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Batch Summary', index=False)

                # Write aggregate statistics
                aggregate_dict = {
                    'total_files': len(results),
                    'successful_files': sum(1 for r in results.values()
                                         if r.get('status') == 'success'),
                    'total_tracks': sum(r.get('n_tracks', 0) for r in results.values()),
                    'total_particles': sum(r.get('n_particles', 0) for r in results.values()),
                    'mean_processing_time': np.mean([r.get('processing_time', 0)
                                                   for r in results.values()]),
                    'total_processing_time': sum(r.get('processing_time', 0)
                                               for r in results.values())
                }

                aggregate_df = pd.DataFrame([aggregate_dict])
                aggregate_df.to_excel(writer, sheet_name='Aggregate Stats', index=False)

                logger.info(f"Saved batch results to {file_path}")
                return True

        except Exception as e:
            logger.error(f"Error writing batch results {file_path}: {str(e)}")
            return False
