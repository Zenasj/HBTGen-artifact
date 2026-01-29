# torch.randn(1025, 2, 256, 256, dtype=torch.float32).cuda()  # inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = (1024, 1024)  # Output dimensions from the issue
        self.mode = "nearest"            # Interpolation mode from the issue
    
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.output_size, mode=self.mode)

def my_model_function():
    return MyModel()

def GetInput():
    # Creates input tensor matching the problematic scenario described in the issue
    return torch.randn(1025, 2, 256, 256, dtype=torch.float32, device="cuda").requires_grad_(True)

