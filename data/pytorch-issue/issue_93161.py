# torch.rand(10, 3, 4), torch.rand(10, 4, 5)  # Input shapes for mat1 and mat2
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        mat1, mat2 = x
        return torch.bmm(mat1, mat2)

def my_model_function():
    return MyModel()

def GetInput():
    mat1 = torch.rand(10, 3, 4)
    mat2 = torch.rand(10, 4, 5)
    return (mat1, mat2)

