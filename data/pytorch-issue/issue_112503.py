# torch.randint(1, 10, (2, 3), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.bitwise_and_(True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(1, 10, (2, 3), dtype=torch.int64)

