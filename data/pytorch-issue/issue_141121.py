# torch.rand(3,3,5,3,5, dtype=torch.float32), torch.rand(3,3,5,3,5, dtype=torch.float32)
import torch
from torch import nn
from torch._inductor import config

full_size = (5, 5, 5, 5, 5)
view_size = (3, 3, 5, 3, 5)

config.triton.max_tiles = 3
config.triton.prefer_nd_tiling = True

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x
        return a + b

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device("cuda")
    full_a = torch.randn(full_size).to(device)
    a = torch.as_strided(full_a, view_size, full_a.stride())
    full_b = torch.randn(full_size).to(device)
    b = torch.as_strided(full_b, view_size, full_b.stride())
    return (a, b)

