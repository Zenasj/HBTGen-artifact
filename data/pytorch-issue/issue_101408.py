# torch.rand(3, dtype=torch.float32)  # Inferred input shape: (3,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = torch.div(x, x)
        x = torch.add(x, 3)
        x = torch.mul(x, x)
        return torch.sub(x, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

