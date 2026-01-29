# torch.rand(1, 1, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp_mode = mode
        self.sf = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.sf, mode=self.interp_mode)

def my_model_function():
    # Initialize with scale_factor=0.5 and mode='area' as per the original issue's setup
    return MyModel(scale_factor=0.5, mode='area')

def GetInput():
    # Generate 4D tensor matching the input shape in the original sample code
    return torch.rand(1, 1, 100, 100, dtype=torch.float32)

