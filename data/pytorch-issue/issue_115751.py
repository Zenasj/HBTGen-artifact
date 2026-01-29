# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input):
        x = input[0]
        a = input[1]
        n = (a * 1.234) // 8.234
        return x + n

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32)

