"""Start the ParticleAnalysis web app on http://127.0.0.1:8129 (local only).

Usage: python run_web.py [--port 8129] [--reload]
"""
import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8129)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    uvicorn.run("particle_analysis.web.app:app", host="127.0.0.1", port=a.port, reload=a.reload,
                reload_dirs=[str(Path(__file__).resolve().parent / "src")] if a.reload else None)


if __name__ == "__main__":
    main()
