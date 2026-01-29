# torch.rand(1)  # Input shape: (1,), float tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        if torch.any(x > 2.0):
            raise RuntimeError(f"Input value {x.item()} exceeds threshold of 2.0")
        return x + 3.0  # Matches the example's behavior of adding 3

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a random float tensor in [0, 5) to occasionally trigger the exception
    return torch.rand(1) * 5.0

