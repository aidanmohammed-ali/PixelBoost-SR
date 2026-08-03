"""
@file pixelboost.py
@author Aidan Mohammed-Ali
@brief Python ctypes wrapper for PixelBoost-SR C shared engine library.
@date 2026-08-03
"""

# =============================
# System Headers
# =============================
import os
import sys
import ctypes
import typing

# =============================
# Optional Helper Libraries
# =============================
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import PIL.Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# =============================
# C Engine Opaque Handle
# =============================
"""
@brief Opaque pointer struct matching PixelBoostEngine in pixelboost.h.
"""
class PixelBoostEngine(ctypes.Structure):
    pass

# =============================
# Main Python Wrapper Class
# =============================
"""
@brief High-level Python interface for the native C Super-Resolution engine.
"""
class PixelBoost:
    """
    @brief Initialise the PixelBoost engine by loading the native C shared library.
    @param model_path Path to the exported .onnx model file.
    @raises FileNotFoundError If libpixelboost.so / pixelboost.dll is missing.
    @raises RuntimeError If the native C engine fails to load the ONNX model.
    """
    def __init__(self, model_path: str):
        # Resolve project root
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        lib_dir = os.path.join(project_root, "lib")
        
        # Select platform-specific dynamic library binary
        if sys.platform.startswith("win"):
            lib_path = os.path.join(lib_dir, "pixelboost.dll")
        else:
            lib_path = os.path.join(lib_dir, "libpixelboost.so")
        
        # Verify library file existence
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Shared library not found at '{lib_path}'. "
                f"Please compile the C shared library using CMake first."
            )
        
        # Load the shared library
        self._lib = ctypes.CDLL(lib_path)
        
        # =============================
        # C API Function Signatures
        # =============================
        
        # PixelBoostEngine *pb_create_engine(const char *model_path);
        self._lib.pb_create_engine.argtypes = [ctypes.c_char_p]
        self._lib.pb_create_engine.restype = ctypes.POINTER(PixelBoostEngine)
        
        # int pb_upscale_image(PixelBoostEngine *engine, const uint8_t *input_rgb, int width, int height, uint8_t *output_rgb);
        self._lib.pb_upscale_image.argtypes = [
            ctypes.POINTER(PixelBoostEngine),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self._lib.pb_upscale_image.restype = ctypes.c_int
        
        # int pb_upscale_image_tiled(PixelBoostEngine *engine, const uint8_t *input_rgb, int width, int height, uint8_t *output_rgb, int tile_size, int overlap);
        self._lib.pb_upscale_image_tiled.argtypes = [
            ctypes.POINTER(PixelBoostEngine),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.pb_upscale_image_tiled.restype = ctypes.c_int
        
        # void pb_destroy_engine(PixelBoostEngine *engine);
        self._lib.pb_destroy_engine.argtypes = [ctypes.POINTER(PixelBoostEngine)]
        self._lib.pb_destroy_engine.restype = None
        
        # Instantiate native C engine handle
        self._engine = self._lib.pb_create_engine(model_path.encode("utf-8"))
        if not self._engine:
            raise RuntimeError(f"Failed to initialize PixelBoost C engine with model: {model_path}")
    
    """
    @brief Destructor to automatically free C memory allocations when object is destroyed.
    """
    def __del__(self):
        if hasattr(self, "_engine") and self._engine and hasattr(self, "_lib"):
            self._lib.pb_destroy_engine(self._engine)
            self._engine = None
    
    """
    @brief Convert a PIL Image object into raw RGB bytes along with dimensions.
    @param img Input PIL Image object.
    @retval Tuple of (raw_bytes, width, height).
    """
    def _pil_to_bytes(self, img: typing.Any) -> typing.Tuple[bytes, int, int]:
        width, height = img.size
        return img.tobytes(), width, height
    
    """
    @brief Upscale an image by 4x using the native C engine.
    @param input_data Image file path (str), PIL Image object, or NumPy array (H, W, 3) uint8.
    @param tiled Set to True for memory-safe tiled inference (recommended), False for full-frame.
    @param tile_size Base tile resolution in pixels (default: 256).
    @param overlap Overlap border padding in pixels to suppress boundary artifacts (default: 16).
    @retval Upscaled PIL Image, NumPy array, or raw bytes matching the input type.
    @raises ImportError If PIL/NumPy are needed but not installed.
    @raises TypeError If input_data is an unsupported type.
    @raises RuntimeError If native C upscaling fails.
    """
    def upscale(
        self,
        input_data: typing.Union[str, typing.Any],
        tiled: bool = True,
        tile_size: int = 256,
        overlap: int = 16,
    ) -> typing.Any:
        is_path = isinstance(input_data, str)
        
        # Standardise input data into raw RGB bytes and extract dimensions
        if is_path:
            if not HAS_PIL:
                raise ImportError("PIL (Pillow) is required to open image file paths. Install with 'pip install Pillow'.")
            pil_img = PIL.Image.open(input_data).convert("RGB")
            img_bytes, width, height = self._pil_to_bytes(pil_img)
        elif HAS_PIL and isinstance(input_data, PIL.Image.Image):
            pil_img = input_data.convert("RGB")
            img_bytes, width, height = self._pil_to_bytes(pil_img)
        elif HAS_NUMPY and isinstance(input_data, np.ndarray):
            if input_data.dtype != np.uint8 or input_data.ndim != 3 or input_data.shape[2] != 3:
                raise ValueError("NumPy array must be uint8 with shape (height, width, 3).")
            height, width, _ = input_data.shape
            img_bytes = input_data.tobytes()
        else:
            raise TypeError("Unsupported input_data type. Expected file path (str), PIL.Image, or NumPy array.")
        
        # Allocate C buffer for 4x upscaled RGB output
        out_width = width * 4
        out_height = height * 4
        out_size = out_width * out_height * 3
        out_buffer = (ctypes.c_uint8 * out_size)()
        
        # Create a ctypes uint8 array pointer from input bytes
        raw_array = (ctypes.c_uint8 * len(img_bytes)).from_buffer_copy(img_bytes)
        in_ptr = ctypes.cast(raw_array, ctypes.POINTER(ctypes.c_uint8))
        
        # Invoke native C function
        if tiled:
            res = self._lib.pb_upscale_image_tiled(
                self._engine, in_ptr, width, height, out_buffer, tile_size, overlap
            )
        else:
            res = self._lib.pb_upscale_image(
                self._engine, in_ptr, width, height, out_buffer
            )
        
        if res != 0:
            raise RuntimeError(f"Native C engine upscaling failed with return code {res}.")
        
        # Format and return output matching input type preference
        raw_out_bytes = bytes(out_buffer)
        if HAS_NUMPY and isinstance(input_data, np.ndarray):
            return np.frombuffer(raw_out_bytes, dtype=np.uint8).reshape((out_height, out_width, 3))
        elif HAS_PIL:
            return PIL.Image.frombytes("RGB", (out_width, out_height), raw_out_bytes)
        else:
            return raw_out_bytes
