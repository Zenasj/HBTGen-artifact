# torch.rand(2, 2048), torch.rand(128, 2048), torch.rand(128)  # input, weight, bias shapes
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        input, w, b = inputs
        return torch.nn.functional.linear(input, w, b)

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.rand(2, 2048, dtype=torch.float32)
    w = torch.rand(128, 2048, dtype=torch.float32)
    b = torch.rand(128, dtype=torch.float32)
    return (input, w, b)

