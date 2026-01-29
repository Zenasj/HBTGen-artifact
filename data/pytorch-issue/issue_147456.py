# torch.rand(3, 2**31, dtype=torch.float32), torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        arg1, arg2 = inputs
        return torch.cholesky_solve(arg1, arg2)

def my_model_function():
    return MyModel()

def GetInput():
    # Original input uses N=2**31 (too large for practical testing), so using 4 here
    arg1 = torch.rand(3, 4, dtype=torch.float32)
    arg2 = torch.rand(3, 3, dtype=torch.float32)
    return (arg1, arg2)

