# torch.rand(1, 512, 2048), torch.rand(1, 2048, 512)  # Input shapes for A and B
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        A, B = inputs
        return A @ B

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.randn(1, 512, 2048).cuda()
    B = torch.randn(1, 2048, 512).cuda()
    return (A, B)

