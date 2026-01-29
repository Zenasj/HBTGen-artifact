# torch.rand(5, 5, 5)  # Inferred input shape from the example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        results = []
        for i in range(4):
            results.append(x[: x.size(0) - i, i : x.size(2), i:3])
        return tuple(results)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 5, 5)

