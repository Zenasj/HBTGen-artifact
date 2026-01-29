# torch.rand(1, 32, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu((self.linear(x) + x) / 2.0)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(32)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 32, dtype=torch.float32)

