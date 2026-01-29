# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # Kernel reduced to 3 to avoid size mismatch
        self.bn = nn.BatchNorm2d(1)  # Proper batch norm with required parameters
        self.conv3d = nn.Conv3d(1, 1, kernel_size=3, padding=1)  # 3D conv with valid parameters

    def forward(self, x):
        # Process 2D convolution and batch norm
        x_2d = self.conv2d(x)
        x_2d = self.bn(x_2d)
        
        # Process 3D convolution on reshaped input (e.g., for 3D compatibility)
        # Example: Add dummy depth dimension for 3D processing
        x_3d = x.unsqueeze(2)  # Convert (B, C, H, W) → (B, C, 1, H, W)
        x_3d = self.conv3d(x_3d)
        
        # Return both outputs for comparison (as per error discussion)
        return x_2d, x_3d

def my_model_function():
    return MyModel()

def GetInput():
    # Return a 4D tensor compatible with 2D conv and 3D conv (via unsqueeze)
    return torch.rand(1, 1, 3, 5, dtype=torch.float32)  # Height=3 to avoid kernel size > input after padding

