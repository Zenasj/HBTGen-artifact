# torch.rand(B, 10, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, downsample=None):
        super(MyModel, self).__init__()
        self.downsample = nn.Sequential(downsample) if downsample is not None else None

    def forward(self, input):
        if self.downsample is not None:
            return self.downsample(input)
        return input

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(nn.Linear(10, 20))

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 4  # Batch size
    return torch.rand(B, 10, dtype=torch.float32)

