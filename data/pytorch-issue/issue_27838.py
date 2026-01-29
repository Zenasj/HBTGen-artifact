# torch.rand(1, 64, 51, 51, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def trim_2d(t, sized):
    return t.narrow(2, 0, sized.size(2)).narrow(3, 0, sized.size(3))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x = F.avg_pool2d(input, 2, ceil_mode=True)
        upscaled = F.interpolate(x, scale_factor=2)
        # input is 51x51, upscaled is 52x52, trim to match
        upscaled = trim_2d(upscaled, input)
        return torch.cat([input, upscaled], dim=1)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 64, 51, 51, dtype=torch.float32)

