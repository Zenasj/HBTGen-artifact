# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Original path (correct behavior)
        y1 = x.t()
        # Simulate compiled path with detached output (buggy behavior)
        y2 = torch.detach(x.t())
        # Compare values and gradient tracking status
        value_match = torch.allclose(y1, y2)
        grad_match = torch.tensor(y1.requires_grad == y2.requires_grad, dtype=torch.bool)
        return value_match & grad_match

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, requires_grad=True)

