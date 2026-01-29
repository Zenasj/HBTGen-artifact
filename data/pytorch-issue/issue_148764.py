# Inputs: (torch.randn(64, 64, dtype=torch.float32, device='cuda'), torch.randn(64, 64, dtype=torch.float32, device='cuda'), torch.arange(-100, 100, 10, dtype=torch.int64, device='cuda'))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, buckets = inputs
        z = torch.mm(x, y)
        return torch.bucketize(z, buckets)

def my_model_function():
    return MyModel()

def GetInput():
    device = "cuda"
    x = torch.randn(64, 64, dtype=torch.float32, device=device)
    y = torch.randn(64, 64, dtype=torch.float32, device=device)
    buckets = torch.arange(-100, 100, 10, dtype=torch.int64, device=device)
    return (x, y, buckets)

