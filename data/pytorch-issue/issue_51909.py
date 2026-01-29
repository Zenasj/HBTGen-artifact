# torch.rand(B, C, H, W, dtype=torch.float32)  # Example: B=1, C=3, H=224, W=224
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Explicitly use float for p parameter to avoid TorchScript type issues
        return F.normalize(x, p=2.0, dim=1, eps=1e-12)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

