# torch.rand(3, 3, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.full_like(x, 1)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(3, 3, dtype=torch.float32)

