# torch.rand(4, dtype=torch.float32), torch.tensor(12.0, dtype=torch.float32)  # input x and scalar a
import torch
import math
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, a = inputs
        a_val = a.item()
        b = math.floor(a_val + 0.5)
        b_rad = math.radians(a_val)
        b_total = b_rad + b
        return x + b_total

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(4, dtype=torch.float32)
    a = torch.tensor(12.0, dtype=torch.float32)
    return (x, a)

