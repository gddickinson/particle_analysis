"""The repo's analysis steps wrapped for the web API: parameters, detection, tracking, features, motion analysis.

Everything here calls the package's own classes (ParticleDetector, ParticleTracker, FeatureCalculator,
DiffusionAnalyzer); defaults are read from their signatures, so the web app and the desktop app share one source.
Coordinates are pixel centres (the detector's convention); physical units appear only in the motion analysis,
which records the pixel size and frame interval it used and where they came from.
"""
import base64
import inspect
import logging
import warnings
from typing import Dict, List, Optional

import numpy as np

from ..analysis.diffusion import DiffusionAnalyzer
from ..core.feature_calculation import FeatureCalculator
from ..core.particle_detection import Particle, ParticleDetector
from ..core.particle_tracking import ParticleTracker, Track


class PipelineError(ValueError):
    """Bad input or parameters (shown to the user as a 400)."""


def _defaults(cls) -> Dict:
    return {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()
            if k != "self" and v.default is not inspect.Parameter.empty}


DEFAULTS = {
    "detector": {k: v for k, v in _defaults(ParticleDetector).items() if k != "exclude_border"},
    "tracker": _defaults(ParticleTracker),
    "features": _defaults(FeatureCalculator),
    "diffusion": _defaults(DiffusionAnalyzer),
}
# allowed ranges for user-supplied parameters (the desktop settings dialog's ranges where it has them)
LIMITS = {"min_sigma": (0.5, 10.0), "max_sigma": (0.5, 10.0), "sigma_steps": (1, 20), "threshold_rel": (0.01, 1.0),
          "min_distance": (1, 50), "max_distance": (0.1, 100.0), "max_gap": (0, 10), "min_track_length": (2, 1000),
          "pixel_size": (1e-4, 100.0), "frame_interval": (1e-5, 1e4)}
INTS = {"sigma_steps", "min_distance", "max_gap", "min_track_length"}


def parse_number(name: str, raw, default):
    """A form value as a number within LIMITS (blank -> default)."""
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        raise PipelineError(f"{name} must be a number")
    if name in INTS:
        if v != int(v):
            raise PipelineError(f"{name} must be a whole number")
        v = int(v)
    lo, hi = LIMITS.get(name, (-np.inf, np.inf))
    if not lo <= v <= hi:
        raise PipelineError(f"{name} must be between {lo} and {hi}")
    return v


def detector_params(form: Dict) -> Dict:
    # (min_sigma and sigma_steps are accepted for completeness; ParticleDetector stores but does not use them)
    return {k: parse_number(k, form.get(k), d) for k, d in DEFAULTS["detector"].items()}


def detect(movie: np.ndarray, params: Dict) -> List[Particle]:
    """ParticleDetector.detect_movie with the given parameters (fit warnings silenced; failures are skipped)."""
    if movie.ndim != 3:
        raise PipelineError(f"expected a (frames, height, width) stack, got shape {movie.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ParticleDetector(**params).detect_movie(movie)


def particles_to_columns(particles: List[Particle]) -> Dict[str, list]:
    """Column-oriented detections, in the detector's order (the JS tracker relies on that order)."""
    cols = {"frame": [], "x": [], "y": [], "intensity": [], "sigma": [], "snr": []}
    for p in particles:
        for k in cols:
            cols[k].append(int(p.frame) if k == "frame" else float(getattr(p, k)))
    return cols


def columns_to_particles(cols: Dict[str, list], frame_size=(0, 0)) -> List[Particle]:
    n = len(cols["frame"])
    get = lambda k, i, d: float(cols[k][i]) if k in cols else d  # noqa: E731
    return [Particle(frame=int(cols["frame"][i]), x=float(cols["x"][i]), y=float(cols["y"][i]),
                     intensity=get("intensity", i, 0.0), sigma=get("sigma", i, 1.5), snr=get("snr", i, 0.0),
                     frame_size=frame_size) for i in range(n)]


def track(particles: List[Particle], params: Optional[Dict] = None) -> List[Track]:
    return ParticleTracker(**(params or DEFAULTS["tracker"])).track_particles(particles)


def tracks_to_json(tracks: List[Track]) -> List[Dict]:
    return [{"id": t.id, "frames": [int(p.frame) for p in t.particles], "x": [float(p.x) for p in t.particles],
             "y": [float(p.y) for p in t.particles]} for t in tracks]


def json_to_tracks(items: List[Dict]) -> List[Track]:
    out = []
    for it in items:
        ps = [Particle(frame=int(f), x=float(x), y=float(y), intensity=0.0, sigma=0.0, snr=0.0, frame_size=(0, 0))
              for f, x, y in zip(it["frames"], it["x"], it["y"])]
        if ps:
            out.append(Track(id=int(it["id"]), particles=ps, start_frame=ps[0].frame, end_frame=ps[-1].frame))
    return out


def _num(v):
    v = float(v)
    return v if np.isfinite(v) else None


def features(tracks: List[Track]) -> List[Dict]:
    """FeatureCalculator.calculate_track_features for each track (None entries dropped), JSON-ready."""
    calc = FeatureCalculator()
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in tracks:
            f = calc.calculate_track_features(t)
            if f is not None:
                d = f.to_dict()
                out.append({k: ([_num(x) for x in v] if k == "msd_values" else
                                int(v) if k == "track_id" else _num(v)) for k, v in d.items()})
    return out


def motion(tracks: List[Track], pixel_size: float, frame_interval: float) -> List[Dict]:
    """DiffusionAnalyzer.analyze_track per track: MSD (um^2 vs s), the four model fits (with BIC) and the motion
    type ("unclassified" when the MSD is too short or the fits fail)."""
    an = DiffusionAnalyzer(pixel_size=pixel_size, frame_interval=frame_interval)
    out = []
    log = logging.getLogger("particle_analysis.analysis.diffusion")
    level = log.level
    log.setLevel(logging.CRITICAL)  # short tracks make the fits fail; they come back as "unclassified"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in tracks:
                r = an.analyze_track(t)
                if not r:
                    continue
                fits = {}
                for model, fr in r["fit_results"].items():
                    fits[model] = {k: ([_num(x) for x in v] if k == "fit_values" else
                                   int(v) if k == "n_params" else _num(v)) for k, v in fr.items()}
                out.append({"track_id": int(t.id), "motion_type": r["motion_type"].value,
                            "time_lags": [_num(x) for x in r["msd_data"]["time_lags"]],
                            "msd": [_num(x) for x in r["msd_data"]["msd_values"]], "fits": fits})
    finally:
        log.setLevel(level)
    return out


def display_frames(movie: np.ndarray, max_side: int = 512, max_frames: int = 200) -> Dict:
    """8-bit display copy of a movie (base64), contrast-stretched to the 0.5-99.8 percentiles of the whole stack and
    binned by an integer factor so the larger side is at most max_side. Display only: analysis uses the raw data."""
    n = min(movie.shape[0], max_frames)
    m = movie[:n].astype(np.float32)
    f = int(np.ceil(max(m.shape[1:]) / max_side))
    if f > 1:
        h, w = (m.shape[1] // f) * f, (m.shape[2] // f) * f
        m = m[:, :h, :w].reshape(n, h // f, f, w // f, f).mean(axis=(2, 4))
    lo, hi = np.percentile(m, [0.5, 99.8])
    u8 = np.clip((m - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return {"n": int(n), "n_total": int(movie.shape[0]), "height": int(u8.shape[1]), "width": int(u8.shape[2]),
            "bin": f, "raw_shape": list(movie.shape), "display_range": [float(lo), float(hi)],
            "data": base64.b64encode(u8.tobytes()).decode("ascii")}
