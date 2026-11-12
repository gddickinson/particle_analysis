# src/particle_analysis/gui/results_export.py
"""Export of the ResultsViewer's results and plots (split out of results_viewer.py).

Tracks and features are in pixels and frames. The MSD and motion-analysis files need physical units: they are written
only when the viewer's pixel size and frame interval are set, and every such file carries them in its columns.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_results(viewer, directory: Path) -> bool:
    """Write tracks.csv, features.csv and, with units set, msd_track_<id>.csv and diffusion_analysis.csv."""
    try:
        results_dir = Path(directory) / "analysis_results"
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / "tracks.csv", 'w') as f:
            f.write("track_id,frame,x_px,y_px,intensity\n")
            for track in viewer.tracks:
                for p in track.particles:
                    f.write(f"{track.id},{p.frame},{p.x},{p.y},{p.intensity}\n")

        with open(results_dir / "features.csv", 'w') as f:
            if viewer.features:
                headers = list(viewer.features[0].to_dict().keys())
                f.write(",".join(headers) + "\n")
                for feature in viewer.features:
                    d = feature.to_dict()
                    f.write(",".join(str(d[h]) for h in headers) + "\n")

        analyzer = viewer.units.analyzer()
        if analyzer is None:
            logger.warning("MSD and motion analysis not exported: " + viewer.units.describe())
            return True
        px, dt = analyzer.pixel_size, analyzer.frame_interval

        if viewer.current_track_id is not None:
            track = next(t for t in viewer.tracks if t.id == viewer.current_track_id)
            time_lags, msd_values = analyzer.calculate_msd(track)
            with open(results_dir / f"msd_track_{track.id}.csv", 'w') as f:
                f.write("time_lag_s,msd_um2,pixel_size_um,frame_interval_s\n")
                for t, m in zip(time_lags, msd_values):
                    f.write(f"{t},{m},{px},{dt}\n")

        with open(results_dir / "diffusion_analysis.csv", 'w') as f:
            f.write("track_id,motion_type,D_normal_um2_per_s,alpha_anomalous,pixel_size_um,frame_interval_s\n")
            for track in viewer.tracks:
                r = analyzer.analyze_track(track)
                fits = r.get('fit_results', {}) if r else {}
                motion = r['motion_type'].value if r else 'unclassified'
                d = fits.get('normal', {}).get('D', '')
                a = fits.get('anomalous', {}).get('alpha', '')
                f.write(f"{track.id},{motion},{d},{a},{px},{dt}\n")
        return True

    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return False


def save_plots(viewer, directory: Path) -> bool:
    """Save the viewer's plots as PNG files under <directory>/plots."""
    try:
        from pyqtgraph.exporters import ImageExporter
        plots_dir = Path(directory) / "plots"
        plots_dir.mkdir(exist_ok=True)
        for widget, name in ((viewer.track_viewer.plot_widget, "track_overview"), (viewer.msd_plot, "msd_analysis"),
                             (viewer.feature_plot, "feature_analysis"), (viewer.motion_plot, "motion_types"),
                             (viewer.diffusion_plot, "diffusion_coefficients")):
            # pyqtgraph's GraphicsScene has no save(); export the plot item as an image
            ImageExporter(widget.getPlotItem()).export(str(plots_dir / f"{name}.png"))
        return True
    except Exception as e:
        logger.error(f"Error saving plots: {str(e)}")
        return False
