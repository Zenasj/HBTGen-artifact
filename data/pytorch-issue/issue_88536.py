# torch.rand(B, 3, 240, 320, dtype=torch.bfloat16, device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Nearest interpolation (works without conversion)
        out_nearest = F.interpolate(x, scale_factor=0.5, mode='nearest')
        # Bilinear interpolation using workaround (convert to float32)
        x_fp32 = x.float()
        out_bilinear = F.interpolate(x_fp32, scale_factor=0.5, mode='bilinear', align_corners=False)
        return out_nearest, out_bilinear.bfloat16()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 240, 320, dtype=torch.bfloat16, device='cuda')

