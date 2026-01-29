# torch.rand(1, 3, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        # Compute float32 interpolation
        out_float = F.interpolate(x, size=(5, 5), mode='bilinear', align_corners=False)
        # Compute float16 interpolation and convert back to float32
        x_half = x.half()
        out_half = F.interpolate(x_half, size=(5, 5), mode='bilinear', align_corners=False)
        out_half_float = out_half.float()
        # Check if the difference is within a tolerance (1e-4)
        diff = torch.abs(out_float - out_half_float)
        # Return True if all elements are within tolerance, else False
        return torch.all(diff < 1e-4)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 1, 1, dtype=torch.float32)

