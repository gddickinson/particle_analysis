"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from particle_analysis.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -------------------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from particle_analysis.simulation import SimulationParams, simulate_recording  # noqa: E402


def test_info_defaults_come_from_the_classes():
    from particle_analysis.core.particle_tracking import ParticleTracker
    d = client.get("/api/info").json()
    assert d["defaults"]["tracker"]["max_distance"] == ParticleTracker().max_distance
    assert d["demo"]["simulated"] is True and d["demo"]["default_threshold"] in d["demo"]["thresholds"]


def test_demo_detections_find_the_simulated_particles():
    movie = client.get("/api/demo/movie").json()
    det = client.get("/api/demo/detections/0.3").json()["detections"]
    hits = total = 0
    for q in movie["truth"]:
        for f, (x, y, vis) in enumerate(zip(q["x"], q["y"], q["visible"])):
            if not vis:
                continue
            total += 1
            d = [np.hypot(det["x"][i] - x, det["y"][i] - y) for i in range(len(det["frame"])) if det["frame"][i] == f]
            hits += bool(d) and min(d) < 1.5
    assert hits / total > 0.95


def test_demo_motion_reports_units():
    m = client.get("/api/demo/motion/0.2").json()
    assert m["pixel_size_um"] == 0.108 and m["units_source"] == "simulation parameters"
    assert len(m["motion"]) == len(m["tracks"]) > 0
    assert {r["motion_type"] for r in m["motion"]} <= {"confined", "normal_diffusion", "directed", "anomalous", None}


def _tiff_bytes(movie, pixel_um=None):
    import tifffile
    buf = io.BytesIO()
    kw = {}
    if pixel_um:
        kw = dict(imagej=True, resolution=(1 / pixel_um, 1 / pixel_um), metadata={"unit": "um"})
    tifffile.imwrite(buf, movie, **kw)
    return buf.getvalue()


def test_upload_detect_reads_pixel_size_and_detects():
    movie, truth = simulate_recording(SimulationParams(n_frames=6, shape=(64, 64), counts={"immobile": 3}))
    r = client.post("/api/detect", files={"file": ("rec.tif", _tiff_bytes(movie, 0.16), "image/tiff")},
                    data={"threshold_rel": "0.3", "max_frames": "4"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_frames_used"] == 4 and d["n_frames_total"] == 6
    assert abs(d["pixel_size_um"] - 0.16) < 1e-6 and "ImageJ" in d["pixel_size_source"]
    assert d["n"] == len(d["detections"]["frame"]) and d["n"] >= 10


def test_upload_rejects_bad_input():
    r = client.post("/api/detect", files={"file": ("x.png", b"123", "image/png")})
    assert r.status_code == 400
    movie, _ = simulate_recording(SimulationParams(n_frames=2, shape=(32, 32), counts={"immobile": 1}))
    r = client.post("/api/detect", files={"file": ("a.tif", _tiff_bytes(movie), "image/tiff")},
                    data={"threshold_rel": "7"})
    assert r.status_code == 400 and "threshold_rel" in r.text


def test_motion_endpoint_matches_demo():
    m = client.get("/api/demo/motion/0.2").json()
    r = client.post("/api/motion", json={"tracks": m["tracks"], "pixel_size": 0.108, "frame_interval": 0.1})
    assert r.status_code == 200
    assert [x["motion_type"] for x in r.json()["motion"]] == [x["motion_type"] for x in m["motion"]]
    assert client.post("/api/motion", json={"tracks": [], "pixel_size": -1}).status_code == 400


def test_engine_fixture_is_current():
    """tests/fixtures/engine_cases.json still matches the Python code (rerun scripts/make_engine_fixture.py)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    new = make_engine_fixture.build()
    old = json.loads(make_engine_fixture.OUT.read_text())
    assert [c["name"] for c in new["cases"]] == [c["name"] for c in old["cases"]]
    assert new["inputs"].keys() == old["inputs"].keys()
    for k in new["inputs"]:
        np.testing.assert_allclose(new["inputs"][k]["x"], old["inputs"][k]["x"], rtol=1e-9)
    for a, b in zip(new["cases"], old["cases"]):
        assert [t["frames"] for t in a["tracks"]] == [t["frames"] for t in b["tracks"]]
        for fa, fb in zip(a["features"], b["features"]):
            for k in ("alpha", "radius_gyration", "straightness", "asymmetry"):
                if fa[k] is None or fb[k] is None:
                    assert fa[k] == fb[k]
                else:
                    assert abs(fa[k] - fb[k]) <= 1e-9 + 1e-9 * abs(fb[k])
