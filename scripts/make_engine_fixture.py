"""Write tests/fixtures/engine_cases.json: Python tracker and feature output that the JS ports must reproduce.

Cases: the demo detections (simulated recording, see web/demo.py) at two thresholds with several tracker settings,
plus small hand-made cases (nearby particles competing for links, gaps, a track that revisits a position).
The JS tests (web/static/*.test.mjs) and tests/test_web.py (fixture is current) read it.

Run: python scripts/make_engine_fixture.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from particle_analysis.web import demo, pipeline  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
TRACKER_SETS = [
    {"max_distance": 5.0, "max_gap": 2, "min_track_length": 3},
    {"max_distance": 2.0, "max_gap": 0, "min_track_length": 2},
    {"max_distance": 12.0, "max_gap": 4, "min_track_length": 5},
]


def _small(frames, xs, ys):
    return {"frame": frames, "x": [float(v) for v in xs], "y": [float(v) for v in ys],
            "intensity": [100.0 + i for i in range(len(frames))]}


SMALL = {
    "nearby": _small([0, 0, 1, 1, 2, 2], [10, 11, 11, 12, 12, 13], [10] * 6),
    "gaps": _small([0, 1, 4, 5, 9, 10, 11], [10, 11, 14, 15, 19, 20, 21], [5, 5, 5, 5, 5, 5, 5]),
    "revisit": _small([0, 1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0], [0] * 7),
}


def case(name, det, params):
    tracks = pipeline.track(pipeline.columns_to_particles(det), params)
    return {"name": name, "tracker": params, "input": name, "tracks": pipeline.tracks_to_json(tracks),
            "features": pipeline.features(tracks)}


def build():
    cases, inputs = [], {}
    for thr in ("0.2", "0.5"):
        det = demo.detections(thr)["detections"]
        det = {k: det[k] for k in ("frame", "x", "y", "intensity")}
        inputs[f"demo threshold {thr}"] = det
        for params in TRACKER_SETS:
            cases.append(case(f"demo threshold {thr}", det, params))
    for name, det in SMALL.items():
        inputs[name] = det
        cases.append(case(name, det, {"max_distance": 2.0, "max_gap": 3, "min_track_length": 2}))
    return {"source": "scripts/make_engine_fixture.py", "inputs": inputs, "cases": cases}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{OUT.relative_to(ROOT)}: {len(data['cases'])} cases, {OUT.stat().st_size / 1e3:.0f} kB")
