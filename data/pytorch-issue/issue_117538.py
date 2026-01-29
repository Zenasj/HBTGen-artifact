# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, scale_factor=2.3):
        super(MyModel, self).__init__()
        # Uses corrected upsample implementation with scale_factor handling
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        return self.upsample(x)

def my_model_function():
    # Returns model instance with non-integer scale factor to trigger scale handling
    return MyModel()

def GetInput():
    # Random input tensor matching expected 4D shape (B, C, H, W)
    return torch.rand(1, 3, 4, 4, dtype=torch.float32)

