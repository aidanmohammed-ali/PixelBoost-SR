/**
 * @file pixelboost.c
 * @author Aidan Mohammed-Ali
 * @brief Native C implementation of PixelBoost-SR engine using ONNX Runtime.
 * @date 2026-07-30
 */

#include "pixelboost.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>

// Concrete implementation of the opaque PixelBoostEngine handle
struct PixelBoostEngine {
    const OrtApi *g_ort;        /**< Global API table containing ONNX Runtime C function pointers */
    OrtEnv *env;                /**< Execution environment for logging and thread pools */
    OrtSession *session;        /**< Loaded ONNX neural network model session */
    OrtMemoryInfo *memory_info; /**< Memory location descriptor (CPU Host memory) */
};

/**
 * @brief Initialises the Super-Resolution engine using an ONNX model file.
 * 
 * Dynamically allocates a new PixelBoostEngine handle, binds to the ONNX Runtime C API,
 * sets up multi-threaded CPU execution, and loads the computation graph from disk.
 *
 * @param model_path Path to the exported .onnx file (e.g. "super_resolution.onnx").
 * @retval Allocated handle on success, or NULL on failure.
 */
PixelBoostEngine *pb_create_engine(const char *model_path) {
    if (!model_path) {
        return NULL;
    }
    
    // Allocate heap memory for engine container struct
    PixelBoostEngine *engine = (PixelBoostEngine*)malloc(sizeof(PixelBoostEngine));
    if (!engine) {
        return NULL;
    }
    
    // Fetch the ONNX Runtime C API function pointer table
    engine->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);    
    if (!engine->g_ort) {
        free(engine);
        return NULL;
    }
    
    // Create the global ONNX execution environment (Warnings-only logging)
    OrtStatus *status = engine->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "PixelBoost", &engine->env);
    if (status != NULL) {
        engine->g_ort->ReleaseStatus(status);
        free(engine);
        return NULL;
    }
    
    // Configure session execution parameters
    OrtSessionOptions *session_options = NULL;
    engine->g_ort->CreateSessionOptions(&session_options);
    
    // Set CPU parallelism
    engine->g_ort->SetIntraOpNumThreads(session_options, 4);
    
    // Attempt to enable NVIDIA CUDA GPU Execution Provider
    OrtCUDAProviderOptions cuda_options;
    memset(&cuda_options, 0, sizeof(OrtCUDAProviderOptions));
    cuda_options.device_id = 0;
    
    OrtStatus *cuda_status = engine->g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options);
    if (cuda_status == NULL) {
        printf("[*] CUDA GPU Execution Provider enabled successfully.\n");
    } else {
        engine->g_ort->ReleaseStatus(cuda_status);
        printf("[*] CUDA GPU unavailable. Falling back to multi-threaded CPU execution.\n");
    }
    
    // Load and parse the neural network from the .onnx file path
    status = engine->g_ort->CreateSession(engine->env, model_path, session_options, &engine->session);
    
    // Release the options builder handle
    engine->g_ort->ReleaseSessionOptions(session_options);
    
    if (status != NULL) {
        printf("[!] Failed to load ONNX model graph from: %s\n", model_path);
        engine->g_ort->ReleaseStatus(status);
        engine->g_ort->ReleaseEnv(engine->env);
        free(engine);
        return NULL;
    }
    
    // Define CPU host memory allocation strategy for tensor input/output buffers
    status = engine->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator,
                                                OrtMemTypeDefault,
                                                &engine->memory_info);
    
    if (status != NULL) {
        engine->g_ort->ReleaseStatus(status);
        engine->g_ort->ReleaseSession(engine->session);
        engine->g_ort->ReleaseEnv(engine->env);
        free(engine);
        return NULL;
    }
    
    printf("[*] PixelBoost C Engine initialised successfully.\n");
    return engine;
}

/**
 * @brief Upscales a raw RGB image buffer by 4x.
 * @param engine Active engine handle.
 * @param input_rgb Pointer to raw byte array [R,G,B, R,G,B, ...] scaled 0-255.
 * @param width Input image width in pixels.
 * @param height Input image height in pixels.
 * @param output_rgb Buffer pre-allocated to size (width * 4 * height * 4 * 3).
 * @retval 0 on success, non-zero on failure.
 */
int pb_upscale_image(PixelBoostEngine *engine,
                     const uint8_t *input_rgb,
                     int width,
                     int height,
                     uint8_t *output_rgb) {
    if (!engine || !input_rgb || !output_rgb) {
        return -1;
    }
    
    // Pre-processing (Interleaved HWC unit8 -> Planar NCHW float)
    size_t input_tensor_size = 1 * 3 * height * width;
    float *float_input = (float*)malloc(input_tensor_size * sizeof(float));
    if (!float_input) {
        return -1;
    }
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx = (h * width + w) * 3 + c;
                int dst_idx = c * (height * width) + (h * width + w);
                float_input[dst_idx] = (float)input_rgb[src_idx] / 255.0f;
            }
        }
    }
    
    // ONNX runtime tensor creation
    int64_t input_shape[] = {1, 3, height, width};
    OrtValue *input_tensor = NULL;
    
    OrtStatus *status = engine->g_ort->CreateTensorWithDataAsOrtValue(
        engine->memory_info,
        float_input,
        input_tensor_size * sizeof(float),
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != NULL) {
        engine->g_ort->ReleaseStatus(status);
        free(float_input);
        return -2;
    }
    
    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};
    OrtValue *output_tensor = NULL;
    
    // Run inference through neural network
    status = engine->g_ort->Run(
        engine->session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );
    
    if (status != NULL) {
        engine->g_ort->ReleaseStatus(status);
        engine->g_ort->ReleaseValue(input_tensor);
        free(float_input);
        return -3;
    }
    
    // Post-processing (Planar NCHW float -> Interleaved HWC unit8)
    float *float_output = NULL;
    engine->g_ort->GetTensorMutableData(output_tensor, (void**)&float_output);
    
    int out_width = width * 4;
    int out_height = height * 4;
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                int src_idx = c * (out_height * out_width) + (h * out_width + w);
                int dst_idx = (h * out_width + w) * 3 + c;
                
                float val = float_output[src_idx] * 255.0f;
                if (val < 0.0f) {
                    val = 0.0f;
                }
                if (val > 255.0f) {
                    val = 255.0f;
                }
                
                output_rgb[dst_idx] = (uint8_t)val;
            }
        }
    }
    
    // Free intermediate allocations & ONNX wrapper values
    engine->g_ort->ReleaseValue(input_tensor);
    engine->g_ort->ReleaseValue(output_tensor);
    free(float_input);
    
    return 0;
}

/**
 * @brief Helper to extract a tile with zero-padding.
 * @param src Pointer to the full master RGB image buffer.
 * @param img_w Master image width in pixels.
 * @param img_h Master image height in pixels.
 * @param x0 Left edge starting coordinate of the tile in the master image (may be negative).
 * @param y0 Top edge starting coordinate of the tile in the master image (may be negative).
 * @param tile_w Padded width of the tile to extract.
 * @param tile_h Padded height of the tile to extract.
 * @param tile_out Output buffer pre-allocated to size (tile_w * tile_h * 3).
 */
static void extract_tile(const uint8_t *src, int img_w, int img_h,
                         int x0, int y0, int tile_w, int tile_h,
                         uint8_t *tile_out) {
    for (int y = 0; y < tile_h; ++y) {
        int src_y = y0 + y;
        if (src_y < 0) {
            src_y = 0;
        }
        if (src_y >= img_h) {
            src_y = img_h - 1;
        }
        
        for (int x = 0; x < tile_w; ++x) {
            int src_x = x0 + x;
            if (src_x < 0) {
                src_x = 0;
            }
            if (src_x >= img_w) {
                src_x = img_w - 1;
            }
            
            int src_idx = (src_y * img_w + src_x) * 3;
            int dst_idx = (y * tile_w + x) * 3;
            memcpy(&tile_out[dst_idx], &src[src_idx], 3);
        }
    }
}

/**
 * @brief Helper to copy the valid inner core of an upscaled tile to the master output.
 * @param tile_sr Pointer to the upscaled RGB tile buffer.
 * @param tile_sr_w Width of the upscaled tile buffer in pixels.
 * @param crop_x Horizontal crop offset within tile_sr to skip edge padding.
 * @param crop_y Vertical crop offset within tile_sr to skip edge padding.
 * @param crop_w Width of the core region to copy.
 * @param crop_h Height of the core region to copy.
 * @param master_out Pointer to the master upscaled output buffer.
 * @param master_w Width of the master output buffer in pixels.
 * @param out_x0 Target horizontal destination coordinate in master_out.
 * @param out_y0 Target vertical destination coordinate in master_out.
 */
static void copy_tile_to_master(const uint8_t *tile_sr, int tile_sr_w,
                                int crop_x, int crop_y, int crop_w, int crop_h,
                                uint8_t *master_out, int master_w,
                                int out_x0, int out_y0) {
    for (int y = 0; y < crop_h; ++y) {
        int dst_y = out_y0 + y;
        int src_y = crop_y + y;
        for (int x = 0; x < crop_w; ++x) {
            int dst_x = out_x0 + x;
            int src_x = crop_x + x;
            
            int src_idx = (src_y * tile_sr_w + src_x) * 3;
            int dst_idx = (dst_y * master_w + dst_x) * 3;
            memcpy(&master_out[dst_idx], &tile_sr[src_idx], 3);
        }
    }
}

/**
 * @brief Upscales a raw RGB image buffer by 4x using a sliding-window tile grid.
 * @details Divides large images into padded tiles to prevent high memory allocation
 *          bursts and Out-Of-Memory (OOM) faults. Edge overlaps are cropped post-inference
 *          to eliminate seam boundary artifacts.
 * @param engine Active engine handle.
 * @param input_rgb Pointer to raw byte array [R,G,B, R,G,B, ...] scaled 0-255.
 * @param width Input image width in pixels.
 * @param height Input image height in pixels.
 * @param output_rgb Buffer pre-allocated to size (width * 4 * height * 4 * 3).
 * @param tile_size Core dimension of square tiles in pixels (e.g., 256).
 * @param overlap Border padding in pixels to absorb tile boundary artifacts (e.g., 16).
 * @retval 0 on success, non-zero on failure.
 */
int pb_upscale_image_tiled(PixelBoostEngine *engine,
                           const uint8_t *input_rgb,
                           int width,
                           int height,
                           uint8_t *output_rgb,
                           int tile_size,
                           int overlap) {
    if (!engine || !input_rgb || !output_rgb) {
        return -1;
    }
    
    const int padded_tile_w = tile_size + 2 * overlap;
    const int padded_tile_h = tile_size + 2 * overlap;
    
    // Allocate temporary buffers for a single padded tile (input and 4x upscaled output)
    uint8_t *tile_in = (uint8_t*)malloc((size_t)padded_tile_w * padded_tile_h * 3);
    uint8_t *tile_out = (uint8_t *)malloc((size_t)padded_tile_w * 4 * padded_tile_h * 4 * 3);
    
    if (!tile_in || !tile_out) {
        free(tile_in);
        free(tile_out);
        return 1;
    }
    
    for (int y = 0; y < height; y += tile_size) {
        for (int x = 0; x < width; x += tile_size) {
            int cur_tile_w = (x + tile_size > width) ? (width - x) : tile_size;
            int cur_tile_h = (y + tile_size > height) ? (height - y) : tile_size;
            
            int x0 = x - overlap;
            int y0 = y - overlap;
            int actual_padded_w = cur_tile_w + 2 * overlap;
            int actual_padded_h = cur_tile_h + 2 * overlap;
            
            // Extract tile with overlap padding
            extract_tile(input_rgb, width, height, x0, y0, actual_padded_w, actual_padded_h, tile_in);
            
            // Run inference on single tile
            if (pb_upscale_image(engine, tile_in, actual_padded_w, actual_padded_h, tile_out) != 0) {
                free(tile_in);
                free(tile_out);
                return 1;
            }
            
            // Crop border padding artifacts and copy core into master output canvas
            int crop_x = overlap * 4;
            int crop_y = overlap * 4;
            int crop_w = cur_tile_w * 4;
            int crop_h = cur_tile_h * 4;
            
            copy_tile_to_master(tile_out, actual_padded_w * 4,
                                crop_x, crop_y, crop_w, crop_h,
                                output_rgb, width * 4,
                                x * 4, y * 4);
        }
    }
    
    free(tile_in);
    free(tile_out);
    return 0;
}

/**
 * @brief Destroys the engine handle and releases memory resources.
 * @param engine Engine handle to destroy.
 */
void pb_destroy_engine(PixelBoostEngine *engine) {
    if (!engine) {
        return;
    }
    
    // Release ONNX Runtime internal resource handles
    if (engine->memory_info) {
        engine->g_ort->ReleaseMemoryInfo(engine->memory_info);
    }
    if (engine->session) {
        engine->g_ort->ReleaseSession(engine->session);
    }
    if (engine->env) {
        engine->g_ort->ReleaseEnv(engine->env);
    }
    
    // Free the host engine structure heap memory
    free(engine);
    printf("[*] PixelBoost C Engine destroyed successfully.\n");
}
