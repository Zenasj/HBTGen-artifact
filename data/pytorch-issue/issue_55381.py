# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Initialize model with fixed seed for reproducibility
    torch.manual_seed(0)
    return MyModel()

def GetInput():
    return torch.rand(1, 64, 32, 32, dtype=torch.float32)

