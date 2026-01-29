# (torch.rand(512, 512, dtype=torch.bfloat16, device='cuda'), torch.rand(512, 512, dtype=torch.bfloat16, device='cuda'))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        a, b = x
        return torch.mm(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    N = 512
    a = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')
    return (a, b)

