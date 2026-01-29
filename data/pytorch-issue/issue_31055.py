# torch.rand(1, 3, 4, 4, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        # Using `size` instead of `scale_factor` to get the simpler graph
        return F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3, 4, 4, dtype=torch.float32)

