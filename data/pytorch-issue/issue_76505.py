# torch.rand(2, 2, dtype=torch.float32, device='cuda'), torch.rand(2, 2, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x.sin() + y.exp()

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2, 2, dtype=torch.float32, device='cuda')
    y = torch.rand(2, 2, dtype=torch.float32, device='cuda')
    return (x, y)

