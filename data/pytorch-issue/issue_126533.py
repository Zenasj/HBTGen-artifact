# torch.rand((), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Create tensor using torch.Tensor() (CPU if default is set to CUDA)
        t1 = torch.Tensor()
        # Create tensor using constructor respecting default device
        t2 = torch.zeros_like(x)
        # Return boolean tensor indicating device mismatch
        return torch.tensor(t1.device != t2.device, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float32)

