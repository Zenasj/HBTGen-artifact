# torch.rand(1, 1, 2, 2, 4, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn
from torch import Tensor

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.module = nn.AdaptiveAvgPool3d(output_size=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(size=(1, 1, 2, 2, 4), requires_grad=True, dtype=torch.float32)

