# Batch Processing Guide

## Overview

This guide covers advanced batch processing capabilities for analyzing multiple recordings efficiently.

## Parallel Processing

### Using Multiple Cores

```python
from particle_analysis.batch import ParallelProcessor

processor = ParallelProcessor(n_workers=4)

# Process directory of files
results = processor.process_directory(
    input_dir="data/",
    pattern="*.tif",
    save_results=True
)
```

### Distributed Processing

```python
from particle_analysis.batch import DistributedProcessor

# Setup distributed processing
processor = DistributedProcessor(
    cluster_type='dask',
    n_workers=8,
    memory_limit='4GB'
)

# Submit batch job
job = processor.submit_batch(
    files=file_list,
    parameters=params,
    output_dir="results/"
)

# Monitor progress
processor.monitor_progress(job)
```

## Data Management

### Batch Configuration

```yaml
# batch_config.yaml
input:
  directory: "data/"
  pattern: "*.tif"
  recursive: true

processing:
  detection:
    min_sigma: 1.0
    max_sigma: 3.0
    threshold: 0.2
  tracking:
    max_distance: 5.0
    max_gap: 2
    min_length: 3

output:
  directory: "results/"
  format: "excel"
  save_visualizations: true
```

### Result Collection

```python
from particle_analysis.batch import ResultCollector

collector = ResultCollector()

# Collect results
summary = collector.collect_results(
    results_dir="results/",
    group_by="condition"
)

# Export summary
collector.export_summary(
    summary,
    "batch_summary.xlsx"
)
```

## Advanced Features

### Condition-based Processing

```python
# Define conditions
conditions = {
    "control": {
        "pattern": "*_ctrl_*.tif",
        "parameters": {"threshold": 0.2}
    },
    "treatment": {
        "pattern": "*_treat_*.tif",
        "parameters": {"threshold": 0.3}
    }
}

# Process with conditions
processor.process_conditions(conditions)
```

### Progress Tracking

```python
from particle_analysis.batch import ProgressTracker

tracker = ProgressTracker()

# Setup callbacks
@tracker.on_file_start
def file_started(filename):
    print(f"Processing {filename}")

@tracker.on_file_complete
def file_completed(filename, results):
    print(f"Completed {filename}")

# Process with tracking
processor.set_tracker(tracker)
processor.process_batch(files)
```

### Error Handling

```python
from particle_analysis.batch import ErrorHandler

handler = ErrorHandler(
    retry_count=3,
    log_errors=True,
    error_dir="error_logs/"
)

# Process with error handling
processor.set_error_handler(handler)
processor.process_batch(files)
```

## Performance Optimization

### Memory Management

```python
from particle_analysis.batch import MemoryManager

# Configure memory management
memory_manager = MemoryManager(
    max_memory='16GB',
    chunk_size='2GB'
)

processor.set_memory_manager(memory_manager)
```

### Processing Strategies

```python
# Chunk-based processing
processor.process_in_chunks(
    movie,
    chunk_size=100,  # frames
    overlap=10
)

# Stream processing
processor.process_stream(
    movie_iterator,
    buffer_size=1000
)
```

## Integration

### Pipeline Integration

```python
from particle_analysis.pipeline import BatchPipeline

# Create pipeline
pipeline = BatchPipeline([
    BatchPreprocessor(),
    BatchDetector(),
    BatchTracker(),
    BatchAnalyzer()
])

# Run batch pipeline
results = pipeline.run_batch(files)
```

### Reporting

```python
from particle_analysis.reporting import BatchReporter

reporter = BatchReporter(
    template="report_template.md",
    plots=['tracks', 'features', 'statistics']
)

# Generate reports
reporter.generate_reports(
    results,
    output_dir="reports/"
)
```