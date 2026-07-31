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
