# Input is a NestedTensor of 3 sequences with shapes (367, 1024), (1245, 1024), (156, 1024) on CUDA and layout=torch.jagged
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Scaling tensor with shape (B, 1, D) for sequence-level broadcasting
        self.scaling_tensor = nn.Parameter(torch.randn(3, 1, 1024, device="cuda"))
    
    def forward(self, x):
        return x * self.scaling_tensor

def my_model_function():
    return MyModel()

def GetInput():
    a1 = torch.rand(367, 1024, device="cuda")
    a2 = torch.rand(1245, 1024, device="cuda")
    a3 = torch.rand(156, 1024, device="cuda")
    return torch.nested.nested_tensor([a1, a2, a3], layout=torch.jagged, device="cuda")

