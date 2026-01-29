# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn([2, 3]))
        self.p2 = nn.Parameter(torch.randn([2, 3]))
    
    def forward(self, x):
        t = self.p1 + x
        out = t / self.p2
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float32)

