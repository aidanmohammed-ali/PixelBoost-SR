"""
@file dataset.py
@author Aidan Mohammed-Ali
@brief Custom PyTorch Dataset loader for Super-Resolution.
@details Loads High-Resolution (HR) ground-truth images from disk, applies 
         spatial augmentations and dynamically generates Low-Resolution (LR) 
         downscaled image pairs for neural network training.
@date 2026-07-27
"""

# =============================
# System Headers
# =============================
import os
import random

# =============================
# Frameworks
# =============================
import PIL.Image as Image
import torch
import torch.utils.data as torch_data
import torchvision.transforms.v2 as transforms

# =============================
# Dataset Definition
# =============================
"""
@brief Dataset class for loading HR images and generating LR pair dynamically.
@details Inherits from torch.utils.data.Dataset.
"""
class SuperResolutionDataset(torch_data.Dataset):
    """
    @brief Constructor for the dataset.
    @param hr_dir Path to directory containing High-Resolution images.
    @param crop_size Dimension of square High-Resolution patch (e.g. 128).
    @param scale_factor Downsample factor (e.g. 4 for 4x super-resolution).
    """
    def __init__(self, hr_dir: str, crop_size: int = 128, scale_factor: int = 4):
        # Initialise
        super().__init__()
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        
        # Scan for images
        self.image_paths = []
        all_files = sorted(os.listdir(hr_dir))
        
        for fname in all_files:
            if fname.endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(hr_dir, fname)
                self.image_paths.append(full_path)
        
        if len(self.image_paths) == 0:
            raise RuntimeError("Error: No images found!")
        
        # Convert to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
    
    """
    @brief Returns total number of images in dataset.
    @retval int Total number of images in dataset.
    """
    def __len__(self):
        return len(self.image_paths)
    
    """
    @brief Fetches image at index, crops a patch and generates LR/HR tensor pair.
    @param idx Image index to read.
    @retval tuple LR/HR tensor pair.
    """
    def __getitem__(self, idx: int):
        # Load image from disk
        img_path = self.image_paths[idx]
        hr_img = Image.open(img_path).convert("RGB")
        width, height = hr_img.size
        
        # Safety check
        if width < self.crop_size or height < self.crop_size:
            new_w = max(width, self.crop_size)
            new_h = max(height, self.crop_size)
            hr_img = hr_img.resize((new_w, new_h), Image.BICUBIC)
            width, height = hr_img.size
        
        # Pick random X and Y coordinates and apply crop
        x = random.randint(0, width - self.crop_size)
        y = random.randint(0, height - self.crop_size)
        
        hr_patch = hr_img.crop((x, y, x + self.crop_size, y + self.crop_size))
        
        # Apply random data augmentations
        if random.random() > 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.random() > 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Create the Low-Res patch by scaling down
        lr_size = int(self.crop_size / self.scale_factor)
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)
        
        # Convert the raw image buffers to 32-bit float tensors
        hr_tensor = self.to_tensor(hr_patch)
        lr_tensor = self.to_tensor(lr_patch)
        
        return lr_tensor, hr_tensor

# =============================
# Main Execution Block
# =============================
if __name__ == "__main__":
    print("Testing Dataset Loader...")
    
    # Initialise the dataset struct
    dataset = SuperResolutionDataset(hr_dir="data/DIV2K_train_HR", crop_size=128, scale_factor=4)
    
    # Grab the first item
    lr_tensor, hr_tensor = dataset[0]
    
    print(f"Total images found: {len(dataset)}")
    print(f"LR Tensor shape:    {lr_tensor.shape}") # Expected: [3, 32, 32]
    print(f"HR Tensor shape:    {hr_tensor.shape}") # Expected: [3, 128, 128]
    print("Success! Tensors are ready for the neural network.")
