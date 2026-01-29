# torch.rand(1, 3, 256, 256, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, recompute_scale_factor=False)

    def forward(self, x):
        return self.upsample(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3, 256, 256, dtype=torch.float32)

