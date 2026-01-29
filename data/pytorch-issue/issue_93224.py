# torch.randint(0, 256, (2,), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.bitwise_xor(x, 18)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 256, (2,), dtype=torch.int32)

