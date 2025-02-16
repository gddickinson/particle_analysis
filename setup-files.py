# setup.py
from setuptools import setup, find_packages

setup(
    name="particle_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "PyQt6>=6.4.0",
        "pyqtgraph>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-qt>=4.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    python_requires=">=3.8",
)

# pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

# requirements/requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-image>=0.18.0
PyQt6>=6.4.0
pyqtgraph>=0.13.0

# requirements/requirements-dev.txt
-r requirements.txt
pytest>=7.0.0
pytest-qt>=4.2.0
pytest-cov>=4.0.0
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0

# .gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.env
.venv
env/
venv/
ENV/
.idea/
.vscode/
