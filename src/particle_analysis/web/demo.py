"""The demo recording for the web app and the static site: simulated by particle_analysis.simulation (default
SimulationParams, seed 7) and analysed by the package's own detector, tracker, feature calculator and diffusion
analyzer. The repo has no microscope recording of its own, so the demo is simulated and labelled as such; its ground
truth (true positions and motion type of every particle) is shown next to the results.

Detection (Gaussian fitting) and the motion analysis (curve fitting) run in Python, so they are precomputed here for a
few detection thresholds; tracking and track features are light and run in the browser (JS ports, tested against
this code).
"""
from functools import lru_cache
from typing import Dict

from ..simulation import SimulationParams, simulate_recording
from . import pipeline

# detection thresholds (threshold_rel) precomputed for the demo; the detector default is one of them
THRESHOLDS = ["0.15", "0.2", "0.3", "0.4", "0.5"]
DESCRIPTION = ("Simulated with particle_analysis.simulation: 14 fluorescent particles (5 diffusive, 4 confined, "
               "3 directed, 2 immobile) drawn as Gaussian spots (sigma 1.5 px) on a camera background with Poisson "
               "and read noise, some blinking for single frames. Not a real recording.")


@lru_cache(maxsize=1)
def recording():
    return simulate_recording(SimulationParams())


def meta() -> Dict:
    movie, truth = recording()
    p = truth["params"]
    return {"id": "demo", "title": "Simulated demo recording", "description": DESCRIPTION, "simulated": True,
            "n_frames": int(movie.shape[0]), "height": int(movie.shape[1]), "width": int(movie.shape[2]),
            "n_particles": len(truth["particles"]), "counts": p["counts"],
            "pixel_size_um": p["pixel_size_um"], "frame_interval_s": p["frame_interval_s"],
            "pixel_size_source": "simulation parameters", "thresholds": THRESHOLDS,
            "default_threshold": str(pipeline.DEFAULTS["detector"]["threshold_rel"])}


def movie_payload() -> Dict:
    movie, truth = recording()
    return {**meta(), "movie": pipeline.display_frames(movie),
            "truth": [{k: q[k] for k in ("id", "motion", "amplitude", "x", "y", "visible")}
                      for q in truth["particles"]],
            "truth_params": truth["params"]}


@lru_cache(maxsize=None)
def detections(threshold: str) -> Dict:
    movie, _ = recording()
    params = {**pipeline.DEFAULTS["detector"], "threshold_rel": float(threshold)}
    parts = pipeline.detect(movie, params)
    return {"threshold_rel": float(threshold), "params": params, "n": len(parts),
            "detections": pipeline.particles_to_columns(parts)}


@lru_cache(maxsize=None)
def motion(threshold: str) -> Dict:
    """Motion analysis of the tracks from the default tracker on the detections at this threshold."""
    d = detections(threshold)
    tracks = pipeline.track(pipeline.columns_to_particles(d["detections"]))
    m = meta()
    return {"threshold_rel": d["threshold_rel"], "tracker": pipeline.DEFAULTS["tracker"],
            "pixel_size_um": m["pixel_size_um"], "frame_interval_s": m["frame_interval_s"],
            "units_source": "simulation parameters", "tracks": pipeline.tracks_to_json(tracks),
            "motion": pipeline.motion(tracks, m["pixel_size_um"], m["frame_interval_s"])}
