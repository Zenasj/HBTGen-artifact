# torch.rand(B, C) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.utils.spectral_norm(torch.nn.Linear(2, 5))

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C = 7, 2  # Batch size and input features
    return torch.randn(B, C)

