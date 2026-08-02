# PixelBoost-SR

A lightweight, high-performance C/C++ inference engine for real-time Super-Resolution image upscaling powered by ONNX Runtime.

## Features
- **Native C Core:** Low-overhead execution wrapper with zero complex language runtime dependencies.
- **Cross-Platform:** Built using CMake with dynamic dependency resolution across Linux, macOS, and Windows.
- **Hardware Fallback:** Automatic detection of CUDA acceleration with thread-parallel CPU execution fallback.
- **RAII C++ Test Harness:** Included test driver for $4\times$ image upscaling.

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/aidanmohammed-ali/PixelBoost-SR.git
cd PixelBoost-SR
```

### 2. Build the Project
CMake will automatically fetch all necessary dependencies (`stb` headers and ONNX Runtime SDK):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 3. Run $4\times$ Upscaling Test
Pass the target ONNX model, input image, and desired output path:
```bash
./build/test_pixelboost models/super_resolution.onnx input.jpg output_4x.png
```

## Project Structure

```text
PixelBoost-SR/
├── CMakeLists.txt                  # Cross-platform CMake build configuration
├── test_main.cpp                   # C++ integration test harness
├── README.md                       # Project documentation
│
├── include/
│   └── pixelboost.h                # Public C API header
│
├── native/
│   └── pixelboost.c                # Engine core & ONNX Runtime integration
│
├── models/
│   ├── super_resolution.onnx       # Exported 4x ONNX model
│   └── super_resolution.onnx.data
│
└── src/                            # PyTorch training & export pipeline
    ├── dataset.py                  # Custom image dataset loaders
    ├── model.py                    # Super-Resolution neural network architecture
    ├── train.py                    # Model training script
    └── export_onnx.py              # PyTorch to ONNX model exporter
```

## Performance & Benchmarks

PixelBoost-SR is optimized for low-memory CPU execution.

* **Memory-Safe Tiled Grid Execution:** Restricts peak RAM allocation to **~4 MB**, preventing OS Out-Of-Memory (`SIGKILL`) crashes on $8\text{K}+$ canvas outputs.
* **Dynamic INT8 Quantization:** Compresses model weights by **73.9%** ($3.40\text{ MB} \to 0.89\text{ MB}$) and speeds up heavy $1920 \times 1280$ grid rendering by **~14.5% (~4.7 seconds saved)**.

**[Read the Full Benchmark Report & Quality Previews (BENCHMARKS.md)](BENCHMARKS.md)**
