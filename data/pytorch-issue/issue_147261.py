# torch.rand(1, 8, 396 * 300, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=1 / 300, mode="linear")

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 8, 396 * 300, dtype=torch.float32)

