# llama.cpp Quick Start Guide

## Key Features

- **Efficient Inference**: Optimized for both CPU and GPU execution
- **Quantization Support**: Works seamlessly with GGUF format quantized models
- **Low Resource Requirements**: Minimal memory footprint compared to full precision models
- **Multi-Platform**: Runs on Linux, macOS, and Windows

## Prerequisites

- Git for cloning the repository
- CMake (3.13 or later) for CPU Build.. for CUDA Build, CMake 3.18 or later is recommended
- A C++17 compatible compiler
- For CUDA support: NVIDIA CUDA Toolkit (nvcc) and cuDNN
- For CPU optimization: OpenBLAS or other BLAS library

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### 2. Build Configuration

Choose the appropriate build for your hardware:

#### CPU-Optimized Build (Recommended for most users)

```bash
cmake -B build-cpu -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build-cpu --config Release
```

#### CUDA GPU Build (For NVIDIA GPUs)

```bash
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release
```

### 3. Verification

After building, locate the executables:
- CPU build: `build-cpu/bin/llama-cli`
- CUDA build: `build-cuda/bin/llama-cli`

## Quick Usage

```bash
./build-cpu/bin/llama-cli -m model.gguf -p "Hello, " -n 128
```

Where:
- `-m`: Path to GGUF format model file
- `-p`: Initial prompt
- `-n`: Number of tokens to generate

## Next Steps

- Download GGUF format models from [Hugging Face](https://huggingface.co)
- Refer to the official [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) for detailed documentation
- Check `docs/` folder for more advanced configuration options
