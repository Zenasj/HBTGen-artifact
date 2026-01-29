# torch.rand(1, 64, 10, 9, 8, dtype=torch.float32, device='cuda', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Both pooling layers are initialized with output_size=0 to trigger the backward crash
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=0)
        self.max_pool = nn.AdaptiveMaxPool3d(output_size=0)
    
    def forward(self, x):
        # Apply both pooling operations and return their summed outputs to trigger gradient computation
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        return avg_out.sum() + max_out.sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 64, 10, 9, 8, dtype=torch.float32, device='cuda', requires_grad=True)

