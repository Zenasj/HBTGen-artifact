# torch.randint(0, 256, (torch.randint(99, 101, (1,)).item(),), dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + x

def my_model_function():
    return MyModel()

def GetInput():
    length = torch.randint(99, 101, (1,)).item()
    return torch.randint(0, 256, (length,), dtype=torch.uint8)

