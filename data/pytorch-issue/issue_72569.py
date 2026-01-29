# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (4, 3, 224, 224)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layernorm = LayerNorm2d(3)
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        x = self.layernorm(x)
        x = checkpoint(self.conv, x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(4, 3, 224, 224, requires_grad=True)

