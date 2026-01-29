# torch.randint(0, 10, (1, 12), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        cumsum = torch.ops.aten.cumsum.default(x, 1); x = None
        return cumsum

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1, 12), dtype=torch.int32)

