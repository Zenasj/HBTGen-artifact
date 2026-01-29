# Input shape: tuple containing tensors of shape (1308), (8, 256), (256, 1308), (8, 1308)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, z, w = inputs
        y = torch.addmm(x, y, z)
        return y.view(-1).sin()

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(1308, requires_grad=True, device='cuda')
    y = torch.randn(8, 256, requires_grad=True, device='cuda')
    z = torch.randn(1308, 256, requires_grad=True, device='cuda').transpose(1, 0)
    w = torch.randn(8, 1308, requires_grad=True, device='cuda')
    return (x, y, z, w)

