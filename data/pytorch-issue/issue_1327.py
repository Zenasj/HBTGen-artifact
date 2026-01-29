import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)

def my_model_function():
    return MyModel()

def GetInput():
    # Default input shape for testing: batch=2, channels=3, 32x32 spatial dims
    return torch.randn(2, 3, 32, 32, dtype=torch.float32)

