# CUDA Vector Addition

A simple CUDA program that demonstrates parallel vector addition using GPU computing.

## Overview

This project implements a basic CUDA kernel for adding two vectors element-wise. It showcases fundamental CUDA programming concepts including:

- Host-device memory management
- CUDA kernel launch configuration
- Memory transfer between host and device
- Thread indexing and block organization

## Files

- `basic.cu` - Main CUDA source code with vector addition kernel
- `Makefile` - Build configuration for compiling the project
- `vector_add` - Compiled executable (generated after build)

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `nvcc` compiler available in PATH

## Building and Running

### Compile the project:
```bash
make
```

### Run the program:
```bash
make run
```

### Clean build artifacts:
```bash
make clean
```

## Program Details

The program performs the following operations:

1. **Host Memory Allocation**: Allocates memory for three float arrays (a, b, c) on the CPU
2. **Data Initialization**: Initializes arrays a and b with test values
3. **Device Memory Allocation**: Allocates corresponding memory on the GPU
4. **Data Transfer**: Copies data from host to device memory
5. **Kernel Launch**: Executes the vector addition kernel on the GPU
6. **Result Transfer**: Copies the result back from device to host
7. **Verification**: Prints the first 25 results to verify correctness
8. **Cleanup**: Frees all allocated memory

### Kernel Configuration

- **Block Size**: 256 threads per block
- **Grid Size**: Calculated as `(N + blockSize - 1) / blockSize` to ensure all elements are processed
- **Vector Size**: 1024 elements

### Sample Output

The program outputs the first 25 vector addition results in the format:
```
a[0] + b[0] = 10.00 + 0.00 = 10.00
a[1] + b[1] = 11.00 + 40.00 = 51.00
a[2] + b[2] = 12.00 + 80.00 = 92.00
...
```

## Learning Objectives

This example demonstrates:
- Basic CUDA memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
- Kernel function definition with `__global__` qualifier
- Thread indexing calculation (`blockIdx.x * blockDim.x + threadIdx.x`)
- Host-device data transfer patterns
- Simple parallel algorithm implementation

## Troubleshooting

- Ensure CUDA Toolkit is properly installed
- Verify GPU is CUDA-compatible
- Check that `nvcc` is in your system PATH
- For compilation errors, ensure all CUDA headers are accessible
