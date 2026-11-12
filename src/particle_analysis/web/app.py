"""FastAPI web app for ParticleAnalysis: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8129, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

Detection and the motion analysis run here in Python (web/pipeline.py, the package's own classes); tracking and the
track features run in the browser (static/tracker.js, static/features.js, tested against the Python code).
"""
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import demo, pipeline
from .recording import read_upload

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "ParticleAnalysis"
MAX_UPLOAD = 500 * 1024 * 1024
MAX_FRAMES = 2000

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary: the pipeline defaults (read from the classes) and the demo recording."""
    return {"title": TITLE, "version": "1", "defaults": pipeline.DEFAULTS, "limits": pipeline.LIMITS,
            "demo": demo.meta(), "max_frames": MAX_FRAMES}


@app.get("/api/demo/movie")
def demo_movie():
    """The simulated demo recording (8-bit display copy) and its ground truth."""
    return demo.movie_payload()


@app.get("/api/demo/detections/{threshold}")
def demo_detections(threshold: str):
    """ParticleDetector on the demo recording, default parameters except threshold_rel."""
    if threshold not in demo.THRESHOLDS:
        raise HTTPException(404, "threshold not precomputed")
    return demo.detections(threshold)


@app.get("/api/demo/motion/{threshold}")
def demo_motion(threshold: str):
    """DiffusionAnalyzer on the default-tracker tracks of the demo detections at this threshold."""
    if threshold not in demo.THRESHOLDS:
        raise HTTPException(404, "threshold not precomputed")
    return demo.motion(threshold)


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), max_frames: str = Form(""), min_sigma: str = Form(""),
                 max_sigma: str = Form(""), sigma_steps: str = Form(""), threshold_rel: str = Form(""),
                 min_distance: str = Form("")):
    """Detect particles in an uploaded TIFF stack with ParticleDetector (runs on this computer)."""
    form = dict(min_sigma=min_sigma, max_sigma=max_sigma, sigma_steps=sigma_steps, threshold_rel=threshold_rel,
                min_distance=min_distance)
    try:
        params = pipeline.detector_params(form)
        n_max = int(pipeline.parse_number("max_frames", max_frames, 100)) if max_frames else 100
        if not 1 <= n_max <= MAX_FRAMES:
            raise pipeline.PipelineError(f"max_frames must be between 1 and {MAX_FRAMES}")
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise pipeline.PipelineError("File too large (limit 500 MB); use the desktop app for bigger files.")
        name = Path(file.filename or "upload.tif").name
        movie, info_ = read_upload(data, name, n_max)
        parts = pipeline.detect(movie, params)
    except pipeline.PipelineError as e:
        raise HTTPException(400, str(e))
    return {"id": "upload", "title": name, "simulated": False, **info_, "params": params, "n": len(parts),
            "movie": pipeline.display_frames(movie), "detections": pipeline.particles_to_columns(parts)}


@app.post("/api/motion")
def motion(body: dict = Body(...)):
    """DiffusionAnalyzer on tracks sent by the browser: {tracks: [{id, frames, x, y}], pixel_size, frame_interval}."""
    try:
        px = pipeline.parse_number("pixel_size", body.get("pixel_size"), pipeline.DEFAULTS["diffusion"]["pixel_size"])
        dt = pipeline.parse_number("frame_interval", body.get("frame_interval"),
                                   pipeline.DEFAULTS["diffusion"]["frame_interval"])
        tracks = pipeline.json_to_tracks(body.get("tracks") or [])
    except (pipeline.PipelineError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"bad request: {e}")
    if len(tracks) > 5000:
        raise HTTPException(400, "too many tracks (limit 5000)")
    return {"pixel_size_um": px, "frame_interval_s": dt, "motion": pipeline.motion(tracks, px, dt)}


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/demo/movie"] + [f"/demo/detections/{t}" for t in demo.THRESHOLDS] + [
    f"/demo/motion/{t}" for t in demo.THRESHOLDS]
