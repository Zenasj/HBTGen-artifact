# torch.rand(3, 4, 5, 6, dtype=torch.float32)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fix: Explicitly set scale to float32 to avoid double type issues
        scale_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.register_buffer('scale', torch.from_numpy(scale_np))
        self.register_buffer('zero_point', torch.tensor([1, 2, 3, 4], dtype=torch.int32))
        self.axis = 1
        self.quant_min = 0
        self.quant_max = 255

    def forward(self, x):
        return torch.fake_quantize_per_channel_affine(
            x, 
            self.scale, 
            self.zero_point, 
            self.axis, 
            self.quant_min, 
            self.quant_max
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4, 5, 6)  # Matches the input shape from original code

