# GPU Optimization Guide

## Overview

This guide covers GPU acceleration for particle detection and tracking, which can significantly improve performance for large datasets.

## Requirements

- CUDA-capable NVIDIA GPU
- CUDA Toolkit 11.0 or later
- cupy package installed
- Minimum 4GB GPU memory (8GB+ recommended)

## Installation

1. Install CUDA Toolkit from NVIDIA website
2. Install cupy:
```bash
pip install cupy-cuda11x  # Replace x with your CUDA version
```

## Implementation

### Enabling GPU Support

```python
from particle_analysis.core.gpu_utils import enable_gpu

# Enable GPU processing
enable_gpu()

# Check GPU status
is_gpu_available()
```

### Optimizing Memory Usage

```python
# Set maximum GPU memory usage
set_gpu_memory_limit(0.8)  # Use 80% of available GPU memory

# Enable memory pooling
enable_memory_pool()
```

### GPU-Accelerated Detection

```python
from particle_analysis.core.particle_detection_gpu import ParticleDetectorGPU

# Create GPU-enabled detector
detector = ParticleDetectorGPU(
    min_sigma=1.0,
    max_sigma=3.0,
    threshold_rel=0.2,
    batch_size=4  # Number of frames to process in parallel
)

# Detect particles
particles = detector.detect_movie(movie)
```

## Best Practices

1. Batch Processing
- Process multiple frames simultaneously
- Optimal batch size depends on GPU memory
- Monitor memory usage

2. Memory Management
- Clear GPU memory between large operations
- Use memory pooling for repetitive operations
- Balance CPU and GPU memory usage

3. Performance Optimization
- Use mixed precision when possible
- Minimize CPU-GPU data transfers
- Pre-allocate GPU memory for large arrays

## Troubleshooting

### Common Issues

1. Out of Memory
```python
# Clear GPU memory
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
```

2. Performance Issues
- Reduce batch size
- Monitor GPU utilization
- Check for memory fragmentation

3. Numerical Precision
- Use appropriate data types
- Monitor numerical stability
- Validate results against CPU implementation

## Benchmarking

```python
from particle_analysis.utils.benchmarking import benchmark_gpu

# Run benchmark
results = benchmark_gpu(
    movie_size=(1000, 512, 512),
    n_particles=100,
    batch_sizes=[1, 2, 4, 8]
)

# Print results
print_benchmark_results(results)
```

## Advanced Topics

### Custom CUDA Kernels

```python
import cupy as cp

# Define custom CUDA kernel
gaussian_kernel = cp.RawKernel(r'''
extern "C" __global__
void gaussian_2d(float* input, float* output, int width, int height) {
    // Kernel implementation
}
''', 'gaussian_2d')

# Use custom kernel
gaussian_kernel(input_gpu, output_gpu, width, height)
```

### Multi-GPU Support

```python
# Select specific GPU
cp.cuda.Device(0).use()

# Distribute work across GPUs
def process_multi_gpu(movie):
    n_gpus = cp.cuda.runtime.getDeviceCount()
    chunks = np.array_split(movie, n_gpus)
    results = []
    
    for i, chunk in enumerate(chunks):
        with cp.cuda.Device(i):
            results.append(process_chunk(chunk))
            
    return combine_results(results)
```