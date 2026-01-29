# torch.rand(2000, 100, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.torch_layernorm = nn.LayerNorm(32, elementwise_affine=False)  # Last dimension is 32

    def forward(self, x):
        # PyTorch implementation
        torch_out = self.torch_layernorm(x)
        
        # Custom implementation
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = (var + 1e-5).sqrt()
        custom_out = (x - mean) / std
        
        # Compute difference metric
        difference = (torch_out - custom_out).abs().mean()
        return difference  # Returns mean absolute difference between outputs

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape matches the original test case (H=2000, W=100, C=32)
    return torch.rand(2000, 100, 32, dtype=torch.float32)

