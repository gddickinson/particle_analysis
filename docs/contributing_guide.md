# Contributing to Particle Analysis

Thank you for your interest in contributing to the Particle Analysis project! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Create a development environment:
```bash
conda create -n particle-analysis-dev python=3.11
conda activate particle-analysis-dev
pip install -r requirements/requirements-dev.txt
```

## Development Guidelines

### Code Style

We follow PEP 8 with these additional guidelines:
- Maximum line length: 88 characters (Black formatter default)
- Use type hints for all function parameters and return values
- Document all classes and functions using NumPy docstring format
- Use descriptive variable names

### Testing

- All new features should include unit tests
- Run tests before submitting:
```bash
pytest tests/
```

- Ensure test coverage remains above 80%
- Include test data if needed

### Git Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make commits with clear messages:
```bash
git commit -m "feat: add new feature description"
```

We use conventional commits:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Test updates
- chore: Maintenance tasks

3. Keep your branch up to date:
```bash
git fetch origin
git rebase origin/main
```

4. Submit a pull request

### Pull Request Process

1. Update documentation for any new features
2. Add or update tests as needed
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Project Structure

```
particle_analysis/
├── docs/               # Documentation
├── src/               # Source code
│   ├── core/          # Core functionality
│   ├── analysis/      # Analysis modules
│   ├── gui/           # GUI components
│   ├── io/            # Input/output handling
│   └── visualization/ # Visualization tools
├── tests/             # Test files
└── requirements/      # Dependencies
```

## Documentation

- Update docs for new features or changes
- Follow documentation style guide
- Include examples when relevant
- Update API reference as needed

## Release Process

1. Version bump in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Submit release PR
5. Tag release after merge

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions for questions
- Contact maintainers for guidance

## Code of Conduct

Please note that this project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.