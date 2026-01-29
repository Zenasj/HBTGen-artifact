# torch.rand(1, 1, 1410, 1280, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed filter as in original code (all ones)
        self.filters = nn.Parameter(torch.ones(1, 1, 67, 67), requires_grad=False)
    
    def forward(self, x):
        return F.conv2d(x, self.filters, padding=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 1410, 1280, dtype=torch.float32)

