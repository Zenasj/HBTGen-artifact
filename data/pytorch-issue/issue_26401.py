# torch.rand(2, dtype=torch.float16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.sum(dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float16, device='cuda')

