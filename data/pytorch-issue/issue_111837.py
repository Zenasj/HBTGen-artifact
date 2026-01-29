# torch.rand(3, dtype=torch.float32)  # Inferred input shape for the model

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        self.linear.requires_grad_(False)  # This performs set __contains__ op by memoizing its search of params.
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    linear = torch.nn.Linear(3, 3)
    return MyModel(linear)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(3, dtype=torch.float32)

