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
mkdir build && cd build
cmake ..
cmake -j
cd ..
```

### 3. Run $4\times$ Upscaling Test
Pass the target ONNX model, input image, desired output path (without extension), and upscale mode. For example:
```bash
./lib/test_pixelboost models/super_resolution.onnx input.jpg output_4x tiled
```

## Project Structure

```text
PixelBoost-SR/
├── CMakeLists.txt                  # Cross-platform CMake build configuration
├── test_main.cpp                   # C++ integration test harness
├── README.md                       # Project documentation
├── BENCHMARKS.md                   # Performance analysis
│
├── include/
│   └── pixelboost.h                # Public C API header
│
├── native/
│   └── pixelboost.c                # Engine core & ONNX Runtime integration
│
├── models/
│   ├── super_resolution.onnx       # Exported 4x ONNX model
│   ├── super_resolution.onnx.data
│   └── super_resolution_int8.onnx  # Quantised ONNX model
│
├── src/                            # PyTorch training & export pipeline
│   ├── dataset.py                  # Custom image dataset loaders
│   ├── model.py                    # Super-Resolution neural network architecture
│   ├── train.py                    # Model training script
│   ├── export_onnx.py              # PyTorch to ONNX model exporter
│   └── quantise.py                 # Quantisation to INT8 for ONNX model
│
└── sample_imgs/                    # Example images before and after upscaling
```

## Performance & Benchmarks

PixelBoost-SR is optimized for low-memory CPU execution.

* **Memory-Safe Tiled Grid Execution:** Restricts peak RAM allocation to **~4 MB**, preventing OS Out-Of-Memory (`SIGKILL`) crashes on $8\text{K}+$ canvas outputs.
* **Dynamic INT8 Quantization:** Compresses model weights by **73.9%** ($3.40\text{ MB} \to 0.89\text{ MB}$) and speeds up heavy $1920 \times 1280$ grid rendering by **~14.5% (~4.7 seconds saved)**.

**[Read the Full Benchmark Report & Quality Previews (BENCHMARKS.md)](BENCHMARKS.md)**

## Citations & Acknowledgements

### Model Training Datasets

The neural network model weights in PixelBoost-SR were trained using datasets by the NTIRE and PIRM Super-Resolution Challenges:

```bibtex
@InProceedings{Agustsson_2017_CVPR_Workshops,
  author    = {Agustsson, Eirikur and Timofte, Radu},
  title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {July},
  year      = {2017}
}

@InProceedings{Timofte_2017_CVPR_Workshops,
  author    = {Timofte, Radu and Agustsson, Eirikur and Van Gool, Luc and Yang, Ming-Hsuan and Zhang, Lei and Lim, Bee and others},
  title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {July},
  year      = {2017}
}

@InProceedings{Timofte_2018_CVPR_Workshops,
  author    = {Timofte, Radu and Gu, Shuhang and Wu, Jiqing and Van Gool, Luc and Zhang, Lei and Yang, Ming-Hsuan and Haris, Muhammad and others},
  title     = {NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2018}
}

@InProceedings{Ignatov_2018_ECCV_Workshops,
  author    = {Ignatov, Andrey and Timofte, Radu and others},
  title     = {PIRM challenge on perceptual image enhancement on smartphones: report},
  booktitle = {European Conference on Computer Vision (ECCV) Workshops},
  month     = {January},
  year      = {2019}
}
```

### Sample Image Attributions

The benchmark and sample preview images are sourced under the [Unsplash License](https://unsplash.com/license):

* **Low-Res Test Image (`lowres.jpg`):** *"Underwater photography of red fish"* on [Unsplash](https://unsplash.com/photos/underwater-photography-of-red-fish-K2RH1QZdLF4).
* **High-Res Test Image (`highres.jpg`):** *"Times Square, New York during daytime"* on [Unsplash](https://unsplash.com/photos/time-square-new-york-during-daytime-TaCk3NspYe0).
