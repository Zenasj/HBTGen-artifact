# torch.rand(3, 3, dtype=torch.float64), torch.rand((), dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, inputs):
        matrix, rcond = inputs
        return torch.linalg.pinv(matrix, rcond=rcond)

def my_model_function():
    return MyModel()

def GetInput():
    matrix = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    rcond = torch.tensor(1e-15, dtype=torch.float64, requires_grad=True)
    return (matrix, rcond)

