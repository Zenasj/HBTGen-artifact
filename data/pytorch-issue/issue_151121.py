# torch.rand(1, 10, dtype=torch.bfloat16) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))

    def forward(self, x):
        return x * torch.tensor(1.0, dtype=torch.bfloat16) * self.param

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 10, dtype=torch.bfloat16)

