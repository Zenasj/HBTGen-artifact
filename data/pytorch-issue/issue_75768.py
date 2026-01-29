# torch.tensor(5)  # Input is a scalar integer tensor (shape ())
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = x.item()  # Convert input tensor to integer
        y = torch.zeros((x, x + 2), device='cuda')
        for i in range(2):
            inp = torch.rand((x, x + i), device='cuda')
            weight = torch.rand((x + 2, x + i), device='cuda')
            bias = torch.rand((x, x + 2), device='cuda')
            linear_out = torch.nn.functional.linear(inp, weight, bias)
            y += torch.sin(linear_out)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(5, device='cuda')

