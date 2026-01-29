# torch.rand(1, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, arg0: torch.Tensor):
        y = arg0.item()
        return torch.zeros(y)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 1-element integer tensor (matches the input shape in the issue)
    return torch.randint(1, 10, (1,), dtype=torch.int64)

