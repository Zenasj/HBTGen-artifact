# torch.rand(B, C, H, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.to(torch.float32)
        return torch.nn.functional.pad(x, (0, 3, 0, 0))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1, 1, 13), dtype=torch.int64)

