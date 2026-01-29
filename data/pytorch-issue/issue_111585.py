# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, z):
        x = z.clone()
        result = torch.vmap(lambda t: t.acos_())(x)
        return x, result  # Return both to check identity

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10)

