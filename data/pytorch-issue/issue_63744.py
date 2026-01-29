# torch.rand(2, 3, dtype=torch.float), torch.rand(3, 2, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return torch.sparse.mm(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randn(2, 3).to_sparse().requires_grad_(True)
    b = torch.randn(3, 2).to_sparse()
    return (a, b)

