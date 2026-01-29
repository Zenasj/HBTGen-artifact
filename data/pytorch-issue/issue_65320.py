# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn
import torch.nn.functional as F
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        # Use adaptive_avg_pool2d to avoid the issue with dynamic kernel size
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 224, 224  # Example input shape
    return torch.rand(B, C, H, W, dtype=torch.float32)

