# Installation Guide

## System Requirements

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- Modern CPU (multi-core recommended)
- Graphics card with OpenGL support
- Operating System:
  - Windows 10 or later
  - macOS 10.14 or later
  - Linux (modern distribution)

## Installation Methods

### Method 1: Using pip (Recommended)

1. Create a new environment:
```bash
conda create -n particle-analysis python=3.11
conda activate particle-analysis
```

2. Install the package:
```bash
pip install particle-analysis
```

### Method 2: From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/particle_analysis.git
cd particle_analysis
```

2. Create environment:
```bash
conda create -n particle-analysis python=3.11
conda activate particle-analysis
```

3. Install in development mode:
```bash
pip install -e .
```

### Method 3: Using Binary Packages

#### Windows
1. Download the latest Windows installer from releases
2. Run the installer
3. Follow the installation wizard

#### macOS
1. Download the latest macOS package
2. Open the package
3. Drag the application to Applications folder

## Dependencies

### Core Dependencies
- numpy
- scipy
- pandas
- scikit-image
- PyQt6
- pyqtgraph

### Optional Dependencies
- GPU support: cupy
- Advanced analysis: statsmodels
- Visualization: matplotlib

## Post-Installation

### Verify Installation
```bash
python -c "import particle_analysis; print(particle_analysis.__version__)"
```

### Configure Settings
1. Run the application once
2. Default settings will be created at:
   - Windows: `%APPDATA%/ParticleAnalysis/`
   - macOS: `~/Library/Application Support/ParticleAnalysis/`
   - Linux: `~/.config/particle_analysis/`

## Troubleshooting

### Common Issues

1. Missing Dependencies
```bash
pip install -r requirements.txt
```

2. OpenGL Issues
- Update graphics drivers
- Install OpenGL dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-opengl
  # macOS
  brew install python-opengl
  ```

3. Qt Issues
- Reinstall PyQt:
  ```bash
  pip uninstall PyQt6
  pip install PyQt6
  ```

### Getting Help

1. Check error messages in:
   - Terminal output
   - Log files in installation directory

2. Report issues:
   - Include system information
   - Provide error messages
   - Describe steps to reproduce

## Updating

### Using pip
```bash
pip install --upgrade particle-analysis
```

### From Source
```bash
git pull origin main
pip install -e .
```

## Uninstallation

### Using pip
```bash
pip uninstall particle-analysis
```

### Complete Cleanup
1. Remove package:
```bash
pip uninstall particle-analysis
```

2. Remove configuration:
```bash
# Windows
rd /s /q "%APPDATA%\ParticleAnalysis"
# macOS/Linux
rm -rf ~/.config/particle_analysis
```

## Advanced Configuration

### Environment Variables
- `PARTICLE_ANALYSIS_CONFIG`: Custom config location
- `PARTICLE_ANALYSIS_DEBUG`: Enable debug output
- `PARTICLE_ANALYSIS_GPU`: Enable GPU support

### Configuration File
```yaml
# config.yaml
processing:
  use_gpu: false
  num_threads: 4
  
visualization:
  default_colormap: viridis
  particle_size: 10
  
analysis:
  default_sigma: 1.5
  max_memory_gb: 8
```