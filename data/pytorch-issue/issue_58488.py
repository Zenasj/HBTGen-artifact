# torch.rand(B, 3, dtype=torch.float)
import torch
import torch.nn as nn

class SumParametrization(nn.Module):
    def forward(self, X, Y):
        return X + Y
    def right_inverse(self, Z):
        return (Z, torch.zeros_like(Z))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)  # Input features=3, output=2
        # Register multi-input parametrization on weight
        torch.nn.utils.parametrize.register_parametrization(
            self.linear, "weight", SumParametrization()
        )

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Random input tensor matching (batch, 3)
    return torch.rand(2, 3, dtype=torch.float)

