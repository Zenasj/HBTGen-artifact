# torch.rand(4, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = x.sin().sin()
        return y

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.ones(4, requires_grad=True)

