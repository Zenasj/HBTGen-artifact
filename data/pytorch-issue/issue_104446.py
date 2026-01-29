# torch.rand(20, 1, 16, 16, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 128, (5, 1))
        self.c2 = weight_norm(nn.Conv2d(128, 512, (5, 1)))

    def forward(self, x):
        a = self.c2(self.c1(x))
        c = self.c2(self.c1(x))
        return a, c

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(20, 1, 16, 16)

