# User Guide

## Overview

The Particle Analysis Tool provides a graphical interface for analyzing particle motion in fluorescence microscopy data. This guide covers the main features and how to use them effectively.

## Getting Started

### Loading Data

1. Click "Open File" or use Ctrl+O
2. Select a TIFF stack file
3. The first frame will be displayed in the viewer

### Analysis Parameters

#### Particle Detection
- Min Sigma: Minimum size of particles (pixels)
- Max Sigma: Maximum size of particles (pixels)
- Threshold: Detection sensitivity (0-1)

#### Tracking
- Max Distance: Maximum distance between frames
- Max Gap: Maximum frames a particle can disappear
- Min Length: Minimum track length

### Analysis Methods

#### Full Image Analysis
1. Click "Analyze Full Image"
2. Wait for analysis to complete
3. Results appear in visualization panels

#### Region of Interest (ROI) Analysis
1. Adjust the red ROI box in the image
2. Click "Detect Particles"
3. Click "Track Particles"
4. View results in visualization panels

### Visualization Panels

#### Image Viewer
- Frame navigation
- Particle overlay toggle
- Intensity level adjustment

#### Track Visualization
- Track display
- Color coding options
- Frame-by-frame playback

#### Feature Analysis
- Feature selection dropdown
- Histogram display
- Statistics panel

#### Results Tables
- Tracks data
- Feature measurements
- Filtering and sorting
- Export options

### Exporting Results

#### Data Export
1. File → Save Results
2. Choose format (Excel/CSV)
3. Select location and save

#### Movie Export
1. File → Export Tracking Movie
2. Set parameters (FPS, duration)
3. Save as video file

### Batch Processing

1. Analysis → Batch Process
2. Add files to process
3. Set output directory
4. Start processing
5. Results saved automatically

## Tips and Tricks

### Optimization
- Start with a small ROI to test parameters
- Use "Show Particles" to verify detection
- Adjust threshold for optimal detection
- Balance max distance with frame rate

### Common Issues
- Too many false positives: Increase threshold
- Missed particles: Decrease threshold
- Broken tracks: Increase max gap
- False connections: Decrease max distance

### Performance
- Large files may take longer to load
- ROI analysis is faster for parameter testing
- Close other applications for better performance
- Use batch processing for multiple files