"""
@file test_python.py
@author Aidan Mohammed-Ali
@brief Test script to verify Python ctypes -> C engine inference.
@date 2026-08-03
"""

# =============================
# System Headers
# =============================
import os

# =============================
# Custom Modules
# =============================
import src.pixelboost as PB

# Paths relative to project root
model_path = "models/super_resolution_int8.onnx"
input_image = "sample_imgs/lowres.jpg"
output_image = "sample_imgs/python_lowres_output_4x.png"

print(f"[*] Initialising PixelBoost engine using: {model_path}")
engine = PB.PixelBoost(model_path)

print(f"[*] Executing 4x super-resolution on: {input_image}")

# Runs tiled upscaling by default
result_img = engine.upscale(input_image, tiled=True, tile_size=256, overlap=16)

print(f"[*] Saving output image to: {output_image}")
result_img.save(output_image)

print("[+] Done! Upscaling completed successfully via Python bindings.")
