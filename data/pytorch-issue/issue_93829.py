# torch.randint(0, 256, size=(), dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('tensor_constant', torch.tensor(4, dtype=torch.uint8))

    def forward(self, v3):
        v7 = torch.neg(v3)
        v4 = torch.lt(v7, self.tensor_constant)
        return v7, v4

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 256, size=(), dtype=torch.uint8)

