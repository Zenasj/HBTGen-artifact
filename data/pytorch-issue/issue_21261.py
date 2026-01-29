# torch.rand(B, S, E, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        contiguous_x = x.contiguous()
        view_shape = (x.size(1), -1)
        view_x = contiguous_x.view(*view_shape)
        return view_x.transpose(0, 1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, dtype=torch.float32)

