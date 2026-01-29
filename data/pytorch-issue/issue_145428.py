# torch.rand(20, 20, dtype=torch.float64, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Test 1: Check physical_neg vs view_neg equivalence
        physical_neg = torch.neg(x)
        view_neg = torch._neg_view(x)
        test1 = torch.allclose(physical_neg, view_neg)
        
        # Test 2: In-place operation on negative view
        x_clone = x.clone()
        original_x = x_clone.clone()
        neg_view = torch._neg_view(x_clone)
        neg_view.add_(1.0)
        expected = -original_x + 1.0  # Expected based on original value before modification
        test2 = torch.allclose(neg_view, expected)
        
        return torch.tensor([test1, test2], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(20, 20, device='cuda', dtype=torch.float64, requires_grad=False)

