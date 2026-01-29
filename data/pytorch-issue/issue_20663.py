# torch.rand(3, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, z):
        a = z[:, 0]  # Extract first column as a 1D tensor
        b = z[:, 1]  # Extract second column as a 1D tensor
        return torch.dot(a, b)  # Compute dot product (equivalent to matmul for vectors)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 2, dtype=torch.float32)

