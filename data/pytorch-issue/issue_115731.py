# torch.rand(1, dtype=torch.float16), torch.empty(0, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x  # x is a tuple of two tensors
        mul_result = torch.mul(a, b)
        div_result = torch.div(a, b)
        return mul_result, div_result

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(1, dtype=torch.float16)
    b = torch.empty(0, dtype=torch.float16)
    return (a, b)

