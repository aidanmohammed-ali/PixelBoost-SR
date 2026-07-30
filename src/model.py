"""
@file model.py
@author Aidan Mohammed-Ali
@brief Neural Network architecture for Super-Resolution model.
@date 2026-07-28
"""

# =============================
# Frameworks
# =============================
import torch
import torch.nn as nn

# =============================
# Super Resolution Model
# =============================
"""
@brief A simple Super-Resolution Neural Network using PixelShuffle.
@details Inherits from torch.nn.Module.
"""
class SuperResolutionModel(nn.Module):
    """
    @brief Constructor for the model.
    @param scale_factor Downsample factor (e.g. 4 for 4x super-resolution).
    @param channels Number of feature channels to use across the the network.
    @param num_residuals Number of residual blocks to use in the body.
    """
    def __init__(self, scale_factor: int = 4, channels: int = 64, num_residuals: int = 8):
        # Initialise
        super().__init__()
        
        # Head: Extract initial features from the RGB image
        self.head = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        
        # Body: Deep feature processing using Residuals
        blocks = []
        for _ in range(num_residuals):
            blocks.append(ResidualBlock(channels))
        
        self.body = nn.Sequential(*blocks)
        
        # Upscaler: PixelShuffle logic to achieve 4x scale
        upscale_blocks = []
        current_scale = scale_factor
        
        while current_scale > 1:
            # Safety check
            if current_scale % 2 != 0:
                raise ValueError("Error: scale_factor must be a power of 2 (2, 4, 8, etc...)")
            
            # Append a 2x scaling block
            upscale_blocks.append(nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=3, padding=1))
            upscale_blocks.append(nn.PixelShuffle(2))
            upscale_blocks.append(nn.ReLU(inplace=True))
            
            # Divide by 2 for the next iteration
            current_scale = current_scale // 2
        
        self.upscaler = nn.Sequential(*upscale_blocks)
                    
        # Tail: Collapse the features back down to a 3-channel RGB image
        self.tail = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    """
    @brief The forward execution pipeline.
    @param x The Low-Res input tensor.
    @retval The High-Res output tensor.
    """
    def forward(self, x: torch.Tensor):
        # Extract initial features
        initial_features = self.head(x)
        
        # Pass through the deep residual body
        x = self.body(initial_features)
        
        # Add original features back before scaling
        x = x + initial_features
        
        # Upscale to 4x resolution
        x = self.upscaler(x)
        
        # Output the final RGB image
        x = self.tail(x)
        return self.sigmoid(x)

# =============================
# Residual Block Helper
# =============================
"""
@brief A simple residual block that bypasses memory to prevent signal degradation.
"""
class ResidualBlock(nn.Module):
    """
    @brief Constructor for the residual blocks.
    @param channels Number of feature channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    """
    @brief The forward execution pipeline for a single block.
    @param x Input tensor.
    @retval Output tensor with residual bypass applied.
    """
    def forward(self, x: torch.Tensor):
        # Save a pointer to the input memory buffer
        residual = x
        
        # Process the features
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Add the original buffer back
        return out + residual

# =============================
# Main Execution Block
# =============================
if __name__ == "__main__":
    print("Testing Neural Network Architecture...")
    
    # Simulate a Low-Res tensor from dataset.py [Batch=1, Channels=3, H=32, W=32]
    fake_lr_image = torch.randn(1, 3, 32, 32)
    
    # Initialise the model
    model = SuperResolutionModel(scale_factor=4, channels=64, num_residuals=8)
    
    # Run the forward pass to test the math and memory routing
    fake_hr_image = model.forward(fake_lr_image)
    
    print(f"Input shape:  {fake_lr_image.shape}")
    print(f"Output shape: {fake_hr_image.shape}")
    print("Success! The architecture compiles and runs perfectly.")
