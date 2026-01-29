# torch.rand(3, 8, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Linear(in_features=8, out_features=4)
        self.module2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preout = self.module(x)
        out = self.module2(preout)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(3, 8, dtype=torch.float32)

