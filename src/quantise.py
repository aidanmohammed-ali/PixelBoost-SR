"""
@file quantise.py
@author Aidan Mohammed-Ali
@brief Quantisation script for PixelBoost-SR ONNX models.
@details Converts FP32 ONNX models to dynamic INT8 quantisation for fast CPU execution.
@date 2026-08-02
"""

# =============================
# System Headers
# =============================
import os
import sys

# =============================
# Frameworks
# =============================
import onnxruntime.quantization as ort_quant

# =============================
# Quantisation Logic
# =============================
def quantise_model(input_path: str, output_path: str) -> None:
    # Reads FP32 ONNX model and exports dynamically quantised INT8 ONNX model
    if not os.path.exists(input_path):
        print(f"[!] Error: Model file '{input_path}' not found.")
        sys.exit(1)
    
    print(f"[!] Loading FP32 model: {input_path}")
    fp32_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"[*] FP32 Model Size: {fp32_size:.2f} MB")
    
    print("[*] Quantising weights to dynamic INT8...")
    ort_quant.quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=ort_quant.QuantType.QUInt8
    )
    
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    compression = (1.0 - (int8_size / fp32_size)) * 100.0
    
    print(f"[+] Successfully saved INT8 model to: {output_path}")
    print(f"[+] INT8 Model Size: {int8_size:.2f} MB ({compression:.1f}% reduction)")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        input_model = sys.argv[1]
    else:
        input_model = "models/super_resolution.onnx"
        
    if len(sys.argv) >= 3:
        output_model = sys.argv[2]
    else:
        output_model = "models/super_resolution_int8.onnx"
    
    quantise_model(input_model, output_model)
