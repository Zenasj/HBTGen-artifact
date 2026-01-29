# torch.rand(1, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resample_filter = nn.Parameter(torch.rand(4, 4))  # Initialize as parameter for proper tracking

    def forward(self, x):
        old_channel = x.shape[1]  # Capture channel count before padding to avoid dynamic shapes
        x = F.pad(x, [1, 1, 1, 1])  # Apply padding
        # Construct weight using static old_channel to prevent unknown shape during export
        weight_shape = [old_channel, 1] + list(self.resample_filter.shape)
        weight = self.resample_filter[None, None].repeat(weight_shape)
        # Use old_channel (static) for groups parameter to ensure ONNX compatibility
        x = F.conv2d(x, weight=weight, padding=1, groups=old_channel)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

