# torch.rand(2, 3, 4, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        return self.sm(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(2, 3, 4, dtype=torch.float32)

