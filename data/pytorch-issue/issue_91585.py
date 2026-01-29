# torch.rand(2, 1, 1, dtype=torch.complex128), torch.rand(2,1,1, dtype=torch.complex128), torch.randint(-32768, 1, (2,1), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        b, A, pivots = inputs
        return torch.lu_solve(b, A, pivots)

def my_model_function():
    return MyModel()

def GetInput():
    b = torch.rand(2, 1, 1, dtype=torch.complex128)
    A = torch.rand(2, 1, 1, dtype=torch.complex128)
    pivots = torch.randint(-32768, 1, (2, 1), dtype=torch.int32)
    return (b, A, pivots)

