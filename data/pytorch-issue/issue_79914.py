# torch.rand(1, 2, dtype=torch.float32).cuda().to_sparse() for each tensor in the input tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        y = a * b  # Element-wise multiplication of sparse tensors
        return torch.sparse.sum(y)  # Sum reduction to trigger backward pass

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(1, 2, dtype=torch.float32).cuda().to_sparse().requires_grad_(True)
    b = torch.rand(1, 2, dtype=torch.float32).cuda().to_sparse().requires_grad_(True)
    return (a, b)  # Returns a tuple of two sparse tensors as inputs

