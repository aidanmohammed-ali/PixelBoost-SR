"""
@file export_onnx.py
@author Aidan Mohammed-Ali
@brief Converts PyTorch model into a portable ONNX binary.
@date 2026-07-30
"""

# =============================
# System Headers
# =============================
import os
import sys

# =============================
# Frameworks
# =============================
import torch

# =============================
# Custom Modules
# =============================
import model

# =============================
# Configuration (Macros)
# =============================
WEIGHTS_PATH = "checkpoint_epoch_50.pth"
ONNX_PATH = "super_resolution.onnx"
SCALE_FACTOR = 4
CHANNELS = 64
NUM_RESIDUALS = 8

# =============================
# Main Export
# =============================
def main():
    # Safety check
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[!] Error: Could not find weights file '{WEIGHTS_PATH}'")
        sys.exit(1)
    
    # Initialise model
    print(f"[*] Initialising model structure (Scale: {SCALE_FACTOR}x, Channels: {CHANNELS}, Residuals: {NUM_RESIDUALS})...")
    sr_model = model.SuperResolutionModel(
        scale_factor=SCALE_FACTOR,
        channels=CHANNELS,
        num_residuals=NUM_RESIDUALS
    )
    print(f"[*] Loading trained weights from '{WEIGHTS_PATH}'...")
    
    # Load state dict directly onto CPU
    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    sr_model.load_state_dict(weights)
    sr_model.eval()
    
    # Create dummy tensor representing [Batch_Size=1, Channels=3, Height=128, Width=128]
    dummy_input = torch.randn(1, 3, 128, 128, requires_grad=False)
    
    # Export model
    print(f"[*] Exporting graph to '{ONNX_PATH}'...")
    torch.onnx.export(
        sr_model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"}
        },
        dynamo=False
    )
    
    # Confirm export
    if os.path.exists(ONNX_PATH):
        file_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        print(f"[*] Export successful! Saved to '{ONNX_PATH}' ({file_size_mb:.2f} MB)")
    else:
        print("[!] Export failed: Output file was not generated.")

if __name__ == "__main__":
    main()
