# (torch.rand(2, 3, device="cuda"), torch.rand(2, 3).bool(), torch.tensor(0.3))  # Input tuple shapes
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, mask, y = inputs
        return torch.masked_fill(x, mask, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2, 3, device="cuda")
    mask = torch.ones_like(x).bool()
    y = torch.tensor(0.3)  # CPU scalar tensor as in the repro
    return (x, mask, y)

