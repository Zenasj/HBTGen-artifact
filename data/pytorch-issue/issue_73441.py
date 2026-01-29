# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        
    def forward(self, x):
        x = self.conv(x)
        x += 1  # In-place addition to trigger functionalization handling
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    return torch.rand(B, 3, 32, 32, dtype=torch.float32)

