# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Core layer causing ONNX export issue
        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        return self.pool(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns random 5D tensor (B,C,D,H,W) compatible with 3D convolution
    return torch.rand(1, 3, 16, 64, 64, dtype=torch.float32)

