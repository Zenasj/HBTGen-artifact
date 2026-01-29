# torch.rand(1, 1, 2048, 2048, dtype=...)  # Inferred input shape from the benchmark script

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, mode="bilinear", antialias=False, scale_factor=0.5):
        super(MyModel, self).__init__()
        self.mode = mode
        self.antialias = antialias
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, mode=self.mode, antialias=self.antialias, scale_factor=self.scale_factor)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 2048, 2048, dtype=torch.float32)

