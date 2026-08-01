/**
 * @file test_main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Cross-platform C++ test harness for PixelBoost-SR engine.
 * @date 2026-07-31
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "pixelboost.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>

/**
 * @brief Main entry point for the PixelBoost-SR integration test harness.
 * 
 * Loads a source image, passes raw RGB bytes through the C engine for 4x 
 * super-resolution inference, and writes the output PNG to disk.
 * 
 * @param argc Command-line argument count.
 * @param argv Command-line argument strings (expects: <onnx_model> <input_image> [output_path.png]).
 * @retval 0 on success, 1 on load/write error, or negative engine error code.
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << "<path_to_onnx_model> <path_to_input_images> [output_path.png]\n";
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = (argc >= 4) ? argv[3] : "output_4x.png";
    
    // Load input image into memory (forcing 3 channels: RGB)
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t *raw_input = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    
    if (!raw_input) {
        std::cerr << "[!] Failed to load input image: " << image_path << "\n";
        return 1;
    }
    
    // Custom RAII deleter to free raw input image memory automatically on exit
    std::unique_ptr<uint8_t, void(*)(void*)> input_rgb(raw_input, [](void *ptr) {
        stbi_image_free(ptr);
    });
    
    std::cout << "[*] Loaded image: " << image_path
              << " (" << width << "x" << height << ", forced 3 channels)\n";
    
    // Initialise the native PixelBoost C engine
    PixelBoostEngine *engine = pb_create_engine(model_path.c_str());
    if (!engine) {
        std::cerr << "[!] Failed to create PixelBoost engine handle.\n";
        return 1;
    }
    
    // Custom RAII deleter to destroy engine handle automatically on exit
    std::unique_ptr<PixelBoostEngine, decltype(&pb_destroy_engine)> engine_guard(engine, pb_destroy_engine);
    
    // Allocate destination buffer for 4x upscale output (4W * 4H * 3 channels)
    const int out_width = width * 4;
    const int out_height = height * 4;
    const size_t output_bytes = static_cast<size_t>(out_width) * out_height * 3;
    
    std::vector<uint8_t> output_rgb(output_bytes);
    
    // Run Super-Resolution inference
    std::cout << "[*] Executing 4x super-resolution inference...\n";
    int status = pb_upscale_image(engine, input_rgb.get(), width, height, output_rgb.data());
    
    if (status == 0) {
        std::cout << "[*] Inference succeeded! Saving 4x output to '" << output_path
                  << "' (" << out_width << "x" << out_height << ")...\n";
        
        int write_result = stbi_write_png(output_path.c_str(), out_width, out_height, 3, output_rgb.data(), 
                                          out_width * 3);
        if (write_result) {
            std::cout << "[*] Output successfully written to disk.\n";
        } else {
            std::cerr << "[!] Failed to write output image file.\n";
            return 1;
        }
    } else {
        std::cerr << "[!] Inference failed with error code: " << status << "\n";
        return status;
    }
    
    return 0;
}
