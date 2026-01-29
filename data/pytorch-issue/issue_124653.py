# torch.rand(200, 200, 200, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        m = 1
        shape = a.shape
        for dim in range(len(shape)):
            view_shape = [1] * (dim + 1)
            view_shape[dim] = -1
            arange_tensor = torch.arange(shape[dim], device=a.device).view(view_shape)
            m = torch.mul(m, arange_tensor)
        term1 = a * m
        term2 = (a + 1) * m
        term3 = (a + 2) * m
        return term1 + term2 + term3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(200, 200, 200, dtype=torch.float32)

