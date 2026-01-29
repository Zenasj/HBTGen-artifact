# torch.rand(B, 10, dtype=torch.float32)
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class MyDataClass:
    __slots__ = ["x", "y"]
    x: int
    y: int

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.x, config.y)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    config = MyDataClass(x=10, y=5)
    return MyModel(config)

def GetInput():
    return torch.rand(2, 10, dtype=torch.float32)

