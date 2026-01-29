# torch.rand(1024, 2, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = torch.roll(x, -1, 0)
        x[0].fill_(0.0)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1024, 2, device="cuda")

