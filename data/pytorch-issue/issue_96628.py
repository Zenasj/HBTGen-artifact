# Input: tuple of 7 tensors with shapes [(32,64,1), (32,32,1), (32,1024,1), (32,32,1), (32,32,1), (32,1185), (32,1,1)], dtype=torch.float32, device='cuda'
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        permute, permute_1, permute_2, permute_3, permute_4, view_3, permute_5 = inputs
        cat = torch.cat([permute, permute_1, permute_2, permute_3, permute_4, permute_5], dim=1)
        return (view_3,)

def my_model_function():
    return MyModel()

def GetInput():
    specs = [
        ((32, 64, 1), (64, 1, 64), torch.float32, 'cuda'),
        ((32, 32, 1), (32, 1, 32), torch.float32, 'cuda'),
        ((32, 1024, 1), (1024, 1, 1024), torch.float32, 'cuda'),
        ((32, 32, 1), (32, 1, 32), torch.float32, 'cuda'),
        ((32, 32, 1), (32, 1, 32), torch.float32, 'cuda'),
        ((32, 1185), (1185, 1), torch.float32, 'cuda'),
        ((32, 1, 1), (1, 1, 1), torch.float32, 'cuda')
    ]
    tensors = []
    for sh, st, dt, dev in specs:
        t = torch.rand(sh, dtype=dt, device=dev)
        t = t.as_strided(sh, st)
        tensors.append(t)
    return tuple(tensors)

