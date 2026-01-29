# torch.rand(5, 10, 1).to(torch.int8, device='cuda'), torch.rand(10, 1).to(torch.int8, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        t = torch.bitwise_and(x, y)
        return torch.clamp_max(t, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(5, 10, 1).to(torch.int8).cuda()
    y = torch.rand(10, 1).to(torch.int8).cuda()
    return (x, y)

