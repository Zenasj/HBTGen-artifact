# Inputs: (torch.rand(B,2,1, dtype=torch.complex128, device='cuda'), torch.rand(B,3,1, dtype=torch.complex128, device='cuda').transpose(1,2).conj())
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        A, B = inputs
        return torch.bmm(A, B)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Example batch size
    A = torch.rand(B, 2, 1, dtype=torch.complex128, device='cuda', requires_grad=False)
    B_base = torch.rand(B, 3, 1, dtype=torch.complex128, device='cuda', requires_grad=False)
    B_tensor = B_base.transpose(1, 2).conj()  # Non-contiguous tensor
    return (A, B_tensor)

