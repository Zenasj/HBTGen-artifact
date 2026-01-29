# torch.rand(3, 0, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute current PyTorch behavior outputs
        mean_orig = torch.mean(x, dim=1, keepdim=True)
        sum_orig = torch.sum(x, dim=1, keepdim=True)
        
        # Desired output shape (same as input shape) per proposal
        desired_shape = x.shape
        
        # Check if shapes match the proposal's expectation
        mean_shape_ok = (mean_orig.shape == desired_shape)
        sum_shape_ok = (sum_orig.shape == desired_shape)
        
        # Return boolean indicating discrepancy between current and proposed behavior
        has_discrepancy = not (mean_shape_ok and sum_shape_ok)
        return torch.tensor([has_discrepancy], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.empty(3, 0, 4)

