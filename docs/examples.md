# Examples

## Basic Usage Examples

### Loading and Analyzing a Single File

```python
from particle_analysis.core.particle_detection import ParticleDetector
from particle_analysis.core.particle_tracking import ParticleTracker
from particle_analysis.core.feature_calculation import FeatureCalculator
from particle_analysis.io.readers import DataReader
from particle_analysis.io.writers import DataWriter

# Load data
reader = DataReader()
movie = reader.read_movie("example.tif")

# Detect particles
detector = ParticleDetector(
    min_sigma=1.0,
    max_sigma=3.0,
    threshold_rel=0.2
)
particles = detector.detect_movie(movie)

# Track particles
tracker = ParticleTracker(
    max_distance=5.0,
    max_gap=2,
    min_track_length=3
)
tracks = tracker.track_particles(particles)

# Calculate features
calculator = FeatureCalculator()
features = [calculator.calculate_track_features(track) for track in tracks]
features = [f for f in features if f is not None]

# Save results
writer = DataWriter()
writer.write_tracks_csv(tracks, "tracks.csv")
writer.write_features_csv(features, "features.csv")
```

### Batch Processing Example

```python
import pathlib
from particle_analysis.core import batch_processor

# Set up batch processor
processor = batch_processor.BatchProcessor(
    input_dir="data_folder",
    output_dir="results_folder",
    file_pattern="*.tif",
    parameters={
        'min_sigma': 1.0,
        'max_sigma': 3.0,
        'threshold_rel': 0.2,
        'max_distance': 5.0,
        'max_gap': 2,
        'min_track_length': 3
    }
)

# Run batch processing
results = processor.process_all()
```

### Custom Analysis Example

```python
from particle_analysis.analysis.diffusion import DiffusionAnalyzer
from particle_analysis.analysis.statistics import TrackAnalyzer

# Analyze diffusion
analyzer = DiffusionAnalyzer()
for track in tracks:
    # Calculate MSD
    time_lags, msd_values = analyzer.calculate_msd(track)
    
    # Classify motion
    motion_type = analyzer.classify_motion(track)
    
    # Print results
    print(f"Track {track.id}:")
    print(f"Motion type: {motion_type}")
    print(f"Diffusion coefficient: {analyzer.fit_diffusion_models(time_lags, msd_values)['normal']['D']}")

# Statistical analysis
track_analyzer = TrackAnalyzer()
labels = track_analyzer.classify_tracks(features)
```

## Visualization Examples

### Creating Track Visualization

```python
from particle_analysis.visualization.plot_utils import TrackVisualizer

# Create visualizer
visualizer = TrackVisualizer()

# Plot tracks
fig, ax = plt.subplots()
visualizer.plot_tracks(
    tracks,
    features,
    color_by='diffusion_coefficient',
    ax=ax,
    show_points=True
)
plt.show()

# Create tracking movie
visualizer.create_track_movie(
    tracks,
    "tracking_movie.mp4",
    fps=10,
    tail_length=10
)
```

### Feature Analysis Plots

```python
from particle_analysis.visualization.plot_utils import TrackVisualizer

visualizer = TrackVisualizer()

# Plot feature distributions
fig = visualizer.plot_track_feature_distributions(
    features,
    feature_names=['diffusion_coefficient', 'straightness', 'radius_gyration']
)
plt.show()

# Plot MSD curves
ax = visualizer.plot_msd_curves(
    features,
    color_by='diffusion_coefficient',
    show_fits=True
)
plt.show()
```

## Advanced Usage Examples

### Custom Feature Calculation

```python
import numpy as np
from particle_analysis.core.feature_calculation import FeatureCalculator

class CustomFeatureCalculator(FeatureCalculator):
    def calculate_custom_feature(self, track):
        """Calculate a custom track feature"""
        positions = track.positions
        times = track.frames
        
        # Calculate custom metric
        custom_metric = np.mean(np.diff(positions, axis=0))
        
        return custom_metric

# Use custom calculator
calculator = CustomFeatureCalculator()
features = [calculator.calculate_track_features(track) for track in tracks]
custom_metrics = [calculator.calculate_custom_feature(track) for track in tracks]
```

### Working with Results

```python
import pandas as pd
import numpy as np

# Load results into pandas
tracks_df = pd.read_csv("tracks.csv")
features_df = pd.read_csv("features.csv")

# Basic statistics
print("Track statistics:")
print(f"Number of tracks: {len(features_df)}")
print(f"Mean track length: {tracks_df.groupby('track_id').size().mean()}")
print(f"Mean diffusion coefficient: {features_df['diffusion_coefficient'].mean()}")

# Group tracks by motion type
motion_types = features_df.groupby('motion_type').size()
print("\nMotion type distribution:")
print(motion_types)

# Advanced analysis
# Calculate correlation between features
correlation = features_df[['diffusion_coefficient', 'straightness', 'radius_gyration']].corr()
print("\nFeature correlations:")
print(correlation)
```