# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Use adaptive_avg_pool3d with valid output_size (2, 2, 2)
        return torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 5D tensor compatible with adaptive_avg_pool3d
    return torch.rand(2, 3, 4, 5, 6)

