/**
 * @file test_main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Cross-platform C++ test harness for PixelBoost-SR engine.
 * @date 2026-07-31
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "pixelboost.h"

// Helper timer class for millisecond profiling
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Main entry point for the PixelBoost-SR integration test harness.
 * 
 * Loads a source image, passes raw RGB bytes through the C engine for 4x 
 * super-resolution inference, and writes the output PNG to disk.
 * 
 * @param argc Command-line argument count.
 * @param argv Command-line argument strings (expects: <onnx_model> <input_image> [output_path.png] [mode: all|tiled|full]).
 * @retval 0 on success, 1 on load/write error, or negative engine error code.
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_input_images> [output_path.png] [mode: all|tiled|full]\n";
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_prefix = (argc >= 4) ? argv[3] : "output_4x";
    const std::string mode = (argc >= 5) ? argv[4] : "all";
    
    std::cout << "====================================================\n";
    std::cout << "        PixelBoost-SR Benchmark & Test Driver       \n";
    std::cout << "====================================================\n";
    
    // Load input image into memory (forcing 3 channels: RGB)
    Timer load_timer;
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t *raw_input = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    
    if (!raw_input) {
        std::cerr << "[!] Failed to load input image: " << image_path << "\n";
        return 1;
    }
    
    double load_time_ms = load_timer.elapsed_ms();
    std::cout << "[*] Loaded Image : " << image_path << " (" << width << "x" << height 
              << " @ 3 RGB channels) in " << std::fixed << std::setprecision(2) << load_time_ms << " ms\n";
    
    // Custom RAII deleter to free raw input image memory automatically on exit
    std::unique_ptr<uint8_t, void(*)(void*)> input_rgb(raw_input, [](void *ptr) {
        stbi_image_free(ptr);
    });
    
    // Initialise the native PixelBoost C engine
    Timer init_timer;
    PixelBoostEngine *engine = pb_create_engine(model_path.c_str());
    if (!engine) {
        std::cerr << "[!] Failed to create PixelBoost engine handle.\n";
        return 1;
    }
    double init_time_ms = init_timer.elapsed_ms();
    
    // Custom RAII deleter to destroy engine handle automatically on exit
    std::unique_ptr<PixelBoostEngine, decltype(&pb_destroy_engine)> engine_guard(engine, pb_destroy_engine);
    
    // Allocate destination buffer for 4x upscale output (4W * 4H * 3 channels)
    const int out_width = width * 4;
    const int out_height = height * 4;
    const size_t output_bytes = static_cast<size_t>(out_width) * out_height * 3;
    
    std::vector<uint8_t> output_rgb(output_bytes);
    
    double full_frame_time_ms = 0.0;
    double tiled_time_ms = 0.0;
    
    // Run Full-Frame Super-Resolution inference
    if (mode == "all" || mode == "full") {
        Timer full_frame_timer;
        std::cout << "\n[*] Executing Full-Frame 4x super-resolution inference...\n";
        int status_full = pb_upscale_image(engine, input_rgb.get(), width, height, output_rgb.data());
        full_frame_time_ms = full_frame_timer.elapsed_ms();
        
        if (status_full == 0) {
            std::string full_path = output_prefix + "_full.png";
            std::cout << "[+] Full-Frame inference succeeded in " << full_frame_time_ms << " ms\n";
            stbi_write_png(full_path.c_str(), out_width, out_height, 3, output_rgb.data(), out_width * 3);
            std::cout << "[*] Saved Full-Frame output to: " << full_path << "\n";
        } else {
            std::cerr << "[!] Full-frame inference failed with error code: " << status_full << "\n";
        }
    }
    
    // Run Tiled Sliding-Window inference (256x256 tile size, 16px overlap)
    if (mode == "all" || mode == "tiled") {
        Timer tiled_timer;
        const int tile_size = 256;
        const int overlap = 16;
        std::cout << "\n[*] Executing Tiled 4x super-resolution inference (" << tile_size << "x" << tile_size << " tiles)...\n";
        int status_tiled = pb_upscale_image_tiled(engine, input_rgb.get(), width, height, output_rgb.data(), tile_size, overlap);
        tiled_time_ms = tiled_timer.elapsed_ms();
        
        if (status_tiled == 0) {
            std::string tiled_path = output_prefix + "_tiled.png";
            std::cout << "[+] Tiled inference succeeded in " << tiled_time_ms << " ms\n";
            stbi_write_png(tiled_path.c_str(), out_width, out_height, 3, output_rgb.data(), out_width * 3);
            std::cout << "[*] Saved Tiled output to: " << tiled_path << "\n";
        } else {
            std::cerr << "[!] Tiled inference failed with error code: " << status_tiled << "\n";
        }
    }
    
    // Performance summary
    std::cout << "\n====================================================\n";
    std::cout << "                 Performance Summary                \n";
    std::cout << "====================================================\n";
    std::cout << " Image Load Time   : " << load_time_ms << " ms\n";
    std::cout << " Engine Init Time  : " << init_time_ms << " ms\n";
    if (full_frame_time_ms > 0) {
        std::cout << " Full-Frame Time   : " << full_frame_time_ms << " ms\n";
    }
    if (tiled_time_ms > 0) {
        std::cout << " Tiled Grid Time   : " << tiled_time_ms << " ms\n";
    }
    std::cout << "====================================================\n";
    
    return 0;
}
