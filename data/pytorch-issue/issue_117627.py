# Inputs: three tensors with shape (1, 2, 1, 4, 9, 7), dtypes (float32, float32, float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        input, other, out = inputs
        return torch.logical_xor(input, other, out=out)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1, 2, 1, 4, 9, 7, dtype=torch.float32)
    y = torch.rand(1, 2, 1, 4, 9, 7, dtype=torch.float32)
    z = torch.rand(1, 2, 1, 4, 9, 7, dtype=torch.float16)
    return (x, y, z)

