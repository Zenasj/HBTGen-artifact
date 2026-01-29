# Input: (torch.rand(2, 1, 3, 4, 2, dtype=torch.float32), torch.rand(4, 4, dtype=torch.float32), torch.randint(-64, 32, (4,), dtype=torch.int32))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, inputs):
        B, LU, pivots = inputs
        return torch.lu_solve(B, LU, pivots)

def my_model_function():
    return MyModel()

def GetInput():
    B = torch.rand(2, 1, 3, 4, 2, dtype=torch.float32)
    LU = torch.rand(4, 4, dtype=torch.float32)
    pivots = torch.randint(-64, 32, (4,), dtype=torch.int32)
    return (B, LU, pivots)

