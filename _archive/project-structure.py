particle_analysis/
│
├── src/
│   ├── particle_analysis/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── particle_detection.py
│   │   │   ├── particle_tracking.py
│   │   │   └── feature_calculation.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── diffusion.py
│   │   │   └── statistics.py
│   │   ├── io/
│   │   │   ├── __init__.py
│   │   │   ├── readers.py
│   │   │   └── writers.py
│   │   ├── visualization/
│   │   │   ├── __init__.py
│   │   │   ├── plot_utils.py
│   │   │   └── viewers.py
│   │   └── gui/
│   │       ├── __init__.py
│   │       ├── main_window.py
│   │       ├── analysis_widget.py
│   │       └── viewers.py
│   │
│   └── scripts/
│       └── run_analysis.py
│
├── tests/
│   ├── __init__.py
│   ├── test_particle_detection.py
│   ├── test_particle_tracking.py
│   ├── test_feature_calculation.py
│   ├── test_diffusion.py
│   └── test_io.py
│
├── examples/
│   ├── single_particle_analysis.py
│   └── batch_processing.py
│
├── docs/
│   ├── api/
│   └── user_guide/
│
├── requirements/
│   ├── requirements.txt
│   └── requirements-dev.txt
│
├── setup.py
├── pyproject.toml
├── README.md
└── .gitignore