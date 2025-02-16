# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Feature or change description

### Changed
- Feature or change description

### Deprecated
- Feature or change description

### Removed
- Feature or change description

### Fixed
- Feature or change description

### Security
- Feature or change description

## [1.0.0] - 2025-02-16

### Added
- Initial release of Particle Analysis tool
- Core particle detection using Gaussian fitting
- Particle tracking with nearest-neighbor linking
- Feature calculation including:
  - Mean Square Displacement (MSD) analysis
  - Diffusion coefficient calculation
  - Track shape analysis
  - Motion classification
- Interactive GUI with:
  - Image viewer with particle overlay
  - Track visualization
  - Feature analysis plots
  - Results tables
- Data export capabilities
- Batch processing support

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

## Guidelines for Maintainers

### Version Numbers
- MAJOR version when making incompatible API changes
- MINOR version when adding functionality in a backward compatible manner
- PATCH version when making backward compatible bug fixes

### Entry Format
- Each change should be a single line
- Group related changes under the appropriate type
- Use imperative, present tense: "Add" not "Added" or "Adds"
- Reference issues and pull requests where possible: "Fix memory leak (#123)"

### Section Types
- Added: New features
- Changed: Changes in existing functionality
- Deprecated: Soon-to-be removed features
- Removed: Now removed features
- Fixed: Any bug fixes
- Security: In case of vulnerabilities

### Release Process
1. Update the Unreleased section
2. Move Unreleased changes to new version section
3. Update version numbers in:
   - pyproject.toml
   - __init__.py
   - documentation
4. Update links at bottom of file
5. Create git tag
6. Create GitHub release

### Links
Links should be maintained at the bottom of this file in the format:
```
[Unreleased]: https://github.com/username/project/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/username/project/releases/tag/v1.0.0
```

## Example Entry Format

### Feature Addition
```markdown
### Added
- Add GPU acceleration for particle detection (#45)
- Add support for .nd2 file format
- Add batch processing capability
```

### Bug Fix
```markdown
### Fixed
- Fix memory leak in track visualization (#123)
- Fix incorrect calculation of diffusion coefficient
- Fix crash when loading large files
```

[Unreleased]: https://github.com/username/particle_analysis/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/username/particle_analysis/releases/tag/v1.0.0