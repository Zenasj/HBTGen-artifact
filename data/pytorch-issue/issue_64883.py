# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # AvgPool3d is known to have non-deterministic CUDA backward implementation
        self.pool = nn.AvgPool3d(3)
    
    def forward(self, x):
        # Forward pass uses AvgPool3d which may trigger determinism warnings
        return self.pool(x)

def my_model_function():
    # Returns a model instance with AvgPool3d to test determinism warnings
    return MyModel()

def GetInput():
    # Returns 5D tensor (B, C, D, H, W) matching AvgPool3d requirements
    # Uses CUDA and requires_grad=True to trigger backward path testing
    return torch.randn(2, 3, 3, 3, 3, dtype=torch.float32, requires_grad=True).cuda()

