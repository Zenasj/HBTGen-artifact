# torch.rand(8, dtype=torch.int64, device='cuda'), torch.rand(3, dtype=torch.int64, device='cuda'), torch.rand(3, dtype=torch.int64, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        out, src, index = inputs
        return out.scatter_(0, index, src)

def my_model_function():
    return MyModel()

def GetInput():
    out = torch.randint(0, 100, (8,), dtype=torch.int64, device='cuda')
    src = torch.randint(0, 10000, (3,), dtype=torch.int64, device='cuda')
    index = torch.randint(0, 8, (3,), dtype=torch.int64, device='cuda')
    return (out, src, index)

