# Particle Analysis Tool

A Python-based application for detecting and tracking particles in fluorescence microscopy recordings. This tool provides a graphical user interface for analyzing particle motion, calculating various features, and visualizing results.

## Features

- Particle detection using Gaussian fitting
- Particle tracking with nearest-neighbor linking
- Feature calculation including:
  - Mean Square Displacement (MSD) analysis
  - Diffusion coefficient calculation
  - Track shape analysis
  - Motion classification
- Interactive visualization of:
  - Particle tracks
  - Feature distributions
  - Analysis results
- Data export capabilities
- Batch processing support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/particle_analysis.git
cd particle_analysis
```

2. Create a conda environment:
```bash
conda create -n particle-analysis python=3.11
conda activate particle-analysis
```

3. Install requirements:
```bash
pip install -r requirements/requirements.txt
```

## Usage

1. Launch the application:
```bash
python src/scripts/run_analysis.py
```

2. Load a microscopy recording file (TIFF stack)
3. Set detection parameters and analyze either:
   - Full image using "Analyze Full Image"
   - Region of interest using "Detect Particles" followed by "Track Particles"
4. View results in the visualization panels
5. Export results as needed

## Documentation

See the `docs` folder for detailed documentation:
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Examples](docs/examples.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.