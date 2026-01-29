# torch.rand(1, 1, 1, 128, dtype=torch.float32), ... (input is a tuple of two tensors)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        stacked = torch.stack([x.float(), y.float()], dim=-1)
        return torch.view_as_complex(stacked)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1, 1, 1, 128, dtype=torch.float32)
    y = torch.rand(1, 1, 1, 128, dtype=torch.float32)
    return (x, y)

