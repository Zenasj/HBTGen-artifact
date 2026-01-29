# torch.rand(B, C, H, W, dtype=torch.float32)  # Input tensor shape (B, C, H, W)
# torch.rand(B, H_out, W_out, 2, dtype=torch.float32)  # Grid tensor shape (B, H_out, W_out, 2)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, inputs):
        input_tensor, grid = inputs  # Unpack input tuple
        return F.grid_sample(input_tensor, grid, align_corners=False)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 3, 224, 224  # Example input dimensions
    input_tensor = torch.rand(B, C, H, W, dtype=torch.float32)
    grid = torch.rand(B, H, W, 2, dtype=torch.float32)  # Grid matches input spatial dimensions
    return (input_tensor, grid)  # Returns tuple of required inputs

