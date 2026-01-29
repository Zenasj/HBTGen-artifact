# torch.rand(10, 3, 20, 20, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(3, 3, 5, 2, 1)

    def forward(self, x):
        x = self.cv1(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(10, 3, 20, 20, dtype=torch.float32)

