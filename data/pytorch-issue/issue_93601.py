# Input shape: (torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), torch.tensor(0.5))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        M, mat1, mat2, beta = inputs
        return torch.addmm(M, mat1, mat2, beta=beta)

def my_model_function():
    return MyModel()

def GetInput():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    beta = torch.tensor(0.5)
    return (M, mat1, mat2, beta)

