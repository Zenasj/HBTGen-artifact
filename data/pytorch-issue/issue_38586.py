# torch.rand(2, 2, dtype=torch.float64)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.mat = nn.Parameter(torch.randn(2, 2, dtype=torch.float64))

    def forward(self, vec):
        return (self.mat @ vec).sum()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    vec = torch.randn(1, dtype=torch.float64).expand(2).requires_grad_(True)
    return vec

