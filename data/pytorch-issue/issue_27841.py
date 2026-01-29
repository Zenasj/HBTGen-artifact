# torch.rand(1, 64, 50, 50, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer required for pixel_shuffle path
        self.conv_ps = nn.Conv2d(64, 64*4, kernel_size=1, bias=False)

    def forward(self, input):
        # Original path (interpolate-based upsampling)
        x_orig = F.avg_pool2d(input, 2, ceil_mode=True)
        upscaled_orig = F.interpolate(x_orig, scale_factor=2)
        orig_output = torch.cat([input, upscaled_orig], dim=1)
        
        # Simple path (summing two avg_pools)
        x_simple = F.avg_pool2d(input, 2)
        y_simple = F.avg_pool2d(input, 2)
        simple_output = x_simple + y_simple
        
        # PixelShuffle path
        x_ps = F.avg_pool2d(input, 2, ceil_mode=True)
        x_ps = self.conv_ps(x_ps)
        upscaled_ps = F.pixel_shuffle(x_ps, 2)
        ps_output = torch.cat([input, upscaled_ps], dim=1)
        
        # Return all outputs to capture all discussed model variants
        return orig_output, simple_output, ps_output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 64, 50, 50, dtype=torch.float32)

