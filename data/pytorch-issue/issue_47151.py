# torch.rand(5, dtype=torch.float32)  # The input shape is inferred to be a 1D tensor of size 5

import torch
import torch.nn as nn
import torch.fx as fx

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(5))
        self.ones = torch.tensor([1.])

    def forward(self, x):
        return torch.dot(self.W, x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(5, dtype=torch.float32)

