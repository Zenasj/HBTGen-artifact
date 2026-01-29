# (torch.rand(100, 100), torch.rand(100, 100))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        mean_x = x.mean(dim=0, keepdim=True)
        mean_y = y.mean(dim=0, keepdim=True)
        mean_diff = (mean_x - mean_y).squeeze().norm(p=2)
        
        offsets_x = x - mean_x
        offsets_y = y - mean_y
        
        central_moment_x = offsets_x.pow(2).mean(dim=0)
        central_moment_y = offsets_y.pow(2).mean(dim=0)
        moment_diff = (central_moment_x - central_moment_y).norm(p=2)
        
        return mean_diff + moment_diff  # Non-inplace addition to avoid gradient issues

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(100, 100, requires_grad=True)
    y = torch.randn(100, 100, requires_grad=True)
    return (x, y)

