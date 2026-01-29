# torch.rand(B, H, W, C, dtype=torch.float32)
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)
    
    def forward(self, x):
        x_permuted = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = F.layer_norm(x_permuted, self.normalized_shape, self.weight, self.bias, self.eps)
        return out.permute(0, 3, 1, 2)  # Back to (B, C, H, W)

class MyModel(nn.Module):
    def __init__(self, n_in=10):
        super(MyModel, self).__init__()
        # Original model components
        self.init_conv = nn.Conv2d(n_in, n_in, 1)
        self.checkpoint_conv = nn.Conv2d(n_in, n_in, 1)
        # LayerNorm branch components
        self.layernorm = LayerNorm2d(n_in)
        self.norm_conv = nn.Conv2d(n_in, n_in, 1)

    def forward(self, x):
        # Original path (permute to B,C,H,W)
        x_orig = x.permute(0, 3, 1, 2)
        x_orig = self.init_conv(x_orig)
        x_orig = checkpoint(self.checkpoint_conv, x_orig)
        
        # LayerNorm path
        x_norm = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        x_norm = self.layernorm(x_norm)
        x_norm = checkpoint(self.norm_conv, x_norm)
        
        return (x_orig, x_norm)

def my_model_function():
    return MyModel(n_in=10)

def GetInput():
    # Returns input matching (B, H, W, C) shape required by MyModel
    return torch.rand(1, 10, 10, 10, dtype=torch.float32)

