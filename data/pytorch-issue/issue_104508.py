# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad3d = nn.ConstantPad3d(padding=1, value=1)  # Requires 5D input (N,C,D,H,W)
        self.pad2d = nn.ConstantPad2d(padding=1, value=1)  # Requires 4D input (N,C,H,W)
        
    def forward(self, x):
        # Apply 3D padding to 5D input
        x = self.pad3d(x)
        # Slice to reduce D dimension and create 4D tensor for 2D padding
        x = x[:, :, 0, :, :]  # Take first slice along D dimension
        # Apply 2D padding to 4D input
        return self.pad2d(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns valid 5D input tensor (N=1, C=2, D=3, H=4, W=5)
    return torch.rand(1, 2, 3, 4, 5)

