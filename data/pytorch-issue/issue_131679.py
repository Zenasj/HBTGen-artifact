# torch.rand(4, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = torch.ops.aten.to(x, dtype=torch.float16, copy=True)
        y.mul_(2)
        return y

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(4, dtype=torch.float32)

