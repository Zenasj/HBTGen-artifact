# torch.rand(10, 10, dtype=torch.float32, device="cuda")  # Inferred input shape

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10, device="cuda") for _ in range(10)])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = checkpoint(self.layers[i], x) + x
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((10, 10), device="cuda", requires_grad=True)

# Example usage:
# model = my_model_function()
# x = GetInput()
# y = model(x)
# y.sum().backward()

