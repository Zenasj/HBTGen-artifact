# torch.rand(100000, 10, dtype=torch.float32)  # Inferred input shape from the provided code

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = nn.Parameter(torch.randn(10, 10))
        self.b = nn.Parameter(torch.randn(10))

    def forward(self, input):
        return torch.addmm(self.b, input, self.w)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(100000, 10, dtype=torch.float32)

