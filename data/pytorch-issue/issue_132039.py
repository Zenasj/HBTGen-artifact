# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute condition number on CPU and GPU, return absolute difference
        cond_cpu = torch.linalg.cond(x.to('cpu'))
        cond_gpu = torch.linalg.cond(x.to('cuda'))
        return torch.abs(cond_cpu - cond_gpu)

def my_model_function():
    # Return model instance comparing CPU/GPU condition number outputs
    return MyModel()

def GetInput():
    # Return random 3x3 tensor as input (matches the matrix shape in the issue)
    return torch.rand(3, 3, dtype=torch.float32)

