/**
 * @file pixelboost.h
 * @author Aidan Mohammed-Ali
 * @brief Public C API for PixelBoost-SR Native Super-Resolution Engine.
 * @date 2026-07-30
 */

#ifndef PIXELBOOST_H
#define PIXELBOOST_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Opaque handle hiding ONNX Runtime implementation details
typedef struct PixelBoostEngine PixelBoostEngine;

/**
 * @brief Initialise the Super-Resolution engine using an ONNX model file.
 * @param model_path Path to the exported .onnx file.
 * @retval Pointer to an allocated engine handle, or NULL on failure.
 */
PixelBoostEngine *pb_create_engine(const char *model_path);

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
                     uint8_t *output_rgb);

/**
 * @brief Destroy the engine handle and release memory resources.
 * @param engine Engine handle to destroy.
 */
void pb_destroy_engine(PixelBoostEngine *engine);

#ifdef __cplusplus
}
#endif

#endif // PIXELBOOST_H
