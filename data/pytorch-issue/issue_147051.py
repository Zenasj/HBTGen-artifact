# (torch.rand(20120, 512, device='cuda'), torch.rand(512, 1536, device='cuda'))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        M, K, N = 20120, 512, 1536
        self.a = nn.Parameter(torch.randn([M, N], device='cuda'))
        
    def forward(self, inputs):
        b, c = inputs
        return torch.addmm(self.a, b, c)

def my_model_function():
    return MyModel()

def GetInput():
    M, K, N = 20120, 512, 1536
    b = torch.randn([M, K], device='cuda')
    c = torch.randn([K, N], device='cuda')
    return (b, c)

