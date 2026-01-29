import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, C, H, W, dtype=torch.float16)  # Input shape: (1, 3, 192, 192)
class MyModel(nn.Module):
    def __init__(self, source_size=192, target_size=112, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        # Precompute grid as part of the model (stored as float16)
        identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
        target_shape = (1, 3, target_size, target_size)
        grid = F.affine_grid(identity, target_shape, align_corners=align_corners)
        self.register_buffer("grid", grid.to(torch.float16))  # Grid stored as float16

    def forward(self, x):
        # Full precision computation (upcast inputs and grid to float32)
        x_full = x.to(torch.float32)
        grid_full = self.grid.to(torch.float32)
        full = F.grid_sample(x_full, grid_full, align_corners=self.align_corners)
        
        # Half precision computation (use original float16 inputs/grid)
        half = F.grid_sample(x, self.grid, align_corners=self.align_corners).to(torch.float32)
        
        # Return absolute error between the two results
        return torch.abs(full - half)

def my_model_function():
    # Initialize with parameters from the original issue
    return MyModel(source_size=192, target_size=112, align_corners=False)

def GetInput():
    # Generate random input matching the required shape/dtype
    return torch.randn(1, 3, 192, 192, dtype=torch.float16, device="cuda")

