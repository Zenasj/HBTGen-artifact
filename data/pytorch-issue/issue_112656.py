# torch.rand(3, 3, dtype=torch.float32), repeated three times in a tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        # Replicate reduction operations causing compilation errors
        sub_init = 1
        for val in inputs:
            sub_init -= val  # Triggers error with torch.compile
        mul_init = 1
        for val in inputs:
            mul_init *= val  # Triggers error with torch.compile
        div_init = 1
        for val in inputs:
            div_init /= val  # Triggers error with torch.compile
        return sub_init, mul_init, div_init

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(3, 3, dtype=torch.float32)
    b = torch.rand(3, 3, dtype=torch.float32)
    c = torch.rand(3, 3, dtype=torch.float32)
    return (a, b, c)

