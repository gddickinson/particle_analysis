# src/particle_analysis/simulation.py
"""Simulated single-particle recordings with known ground truth (for demos and tests).

A recording is a (t, y, x) uint16 stack: a flat camera background, every visible particle drawn as a 2D Gaussian
spot (the PSF), then Poisson shot noise and Gaussian read noise. Each particle has one motion type:

- ``diffusive``: Brownian motion with diffusion coefficient D (um^2/s)
- ``confined``: Brownian motion inside a disc of radius ``confinement_radius_um`` (steps leaving it are rejected)
- ``directed``: drift at ``velocity_um_s`` in a fixed random direction, plus diffusion with D
- ``immobile``: fixed position

Particles can blink (miss single frames) to exercise gap closing. Coordinate convention: pixel centres, as the
detector reports them (pixel (row i, column j) has its centre at y = i, x = j). Physical units are converted with
``pixel_size_um`` and ``frame_interval_s``, which are stored with the ground truth.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np

MOTION_TYPES = ("diffusive", "confined", "directed", "immobile")


@dataclass
class SimulationParams:
    """Parameters of a simulated recording (defaults: the web demo)."""
    n_frames: int = 60
    shape: Tuple[int, int] = (128, 128)          # (height, width) in pixels
    pixel_size_um: float = 0.108                 # DiffusionAnalyzer default
    frame_interval_s: float = 0.1                # DiffusionAnalyzer default
    psf_sigma_px: float = 1.5
    background: float = 100.0                    # camera counts
    amplitude: Tuple[float, float] = (110.0, 260.0)  # peak counts above background, uniform range per particle
    read_noise: float = 3.0                      # counts (Gaussian), on top of Poisson shot noise
    blink_probability: float = 0.03              # chance a particle is invisible in a given frame
    margin_px: float = 8.0                       # particles stay at least this far from the border (reflected)
    counts: Dict[str, int] = field(default_factory=lambda: {
        "diffusive": 5, "confined": 4, "directed": 3, "immobile": 2})
    diffusion_um2_s: float = 0.08
    confined_diffusion_um2_s: float = 0.08
    confinement_radius_um: float = 0.35
    velocity_um_s: float = 0.6
    directed_diffusion_um2_s: float = 0.01
    seed: int = 7


def _start_positions(n: int, p: SimulationParams, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) start positions (x, y): a jittered grid inside the margin, so spots rarely start overlapping."""
    h, w = p.shape
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    cw, ch = (w - 2 * p.margin_px) / cols, (h - 2 * p.margin_px) / rows
    cells = rng.permutation(rows * cols)[:n]
    jitter = rng.uniform(-0.25, 0.25, (n, 2))
    return np.column_stack([p.margin_px + (cells % cols + 0.5 + jitter[:, 0]) * cw,
                            p.margin_px + (cells // cols + 0.5 + jitter[:, 1]) * ch])


def _trajectory(kind: str, start: np.ndarray, p: SimulationParams, rng: np.random.Generator) -> np.ndarray:
    """(n_frames, 2) positions (x, y) in pixels for one particle."""
    h, w = p.shape
    dt, px = p.frame_interval_s, p.pixel_size_um
    xy = np.empty((p.n_frames, 2))
    xy[0] = start
    if kind == "immobile":
        xy[:] = start
        return xy
    D = {"diffusive": p.diffusion_um2_s, "confined": p.confined_diffusion_um2_s,
         "directed": p.directed_diffusion_um2_s}[kind]
    step_sd = np.sqrt(2 * D * dt) / px  # per axis, pixels
    drift = np.zeros(2)
    if kind == "directed":
        angle = rng.uniform(0, 2 * np.pi)
        drift = p.velocity_um_s * dt / px * np.array([np.cos(angle), np.sin(angle)])
    radius_px = p.confinement_radius_um / px
    for t in range(1, p.n_frames):
        prop = xy[t - 1] + drift + rng.normal(0, step_sd, 2)
        if kind == "confined" and np.hypot(*(prop - start)) > radius_px:
            prop = xy[t - 1]
        # reflect at the image border (keep the spot inside the field of view)
        for k, size in ((0, w), (1, h)):
            lo, hi = p.margin_px, size - 1 - p.margin_px
            if prop[k] < lo:
                prop[k] = 2 * lo - prop[k]
            elif prop[k] > hi:
                prop[k] = 2 * hi - prop[k]
        xy[t] = prop
    return xy


def simulate_recording(params: SimulationParams = None) -> Tuple[np.ndarray, Dict]:
    """Simulate a recording. Returns (movie uint16 (t, y, x), ground truth dict).

    Ground truth: ``particles`` (list of {id, motion, amplitude, x[], y[], visible[]}) and ``params``.
    """
    p = params or SimulationParams()
    rng = np.random.default_rng(p.seed)
    h, w = p.shape
    kinds: List[str] = [k for k in MOTION_TYPES for _ in range(p.counts.get(k, 0))]
    particles = []
    starts = _start_positions(len(kinds), p, rng)
    for pid, kind in enumerate(kinds):
        xy = _trajectory(kind, starts[pid], p, rng)
        visible = rng.random(p.n_frames) >= p.blink_probability
        visible[0] = visible[-1] = True
        particles.append({"id": pid, "motion": kind, "amplitude": float(rng.uniform(*p.amplitude)),
                          "x": xy[:, 0], "y": xy[:, 1], "visible": visible})
    yy, xx = np.mgrid[0:h, 0:w]
    movie = np.empty((p.n_frames, h, w), dtype=np.uint16)
    two_s2 = 2 * p.psf_sigma_px ** 2
    for t in range(p.n_frames):
        expected = np.full((h, w), p.background)
        for q in particles:
            if q["visible"][t]:
                expected += q["amplitude"] * np.exp(-((xx - q["x"][t]) ** 2 + (yy - q["y"][t]) ** 2) / two_s2)
        frame = rng.poisson(expected) + rng.normal(0, p.read_noise, (h, w))
        movie[t] = np.clip(np.round(frame), 0, 65535).astype(np.uint16)
    truth = {
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
        "particles": [{**q, "x": q["x"].tolist(), "y": q["y"].tolist(), "visible": q["visible"].tolist()}
                      for q in particles],
    }
    return movie, truth
