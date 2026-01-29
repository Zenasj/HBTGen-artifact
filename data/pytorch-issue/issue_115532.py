# torch.rand(B, C, H, W, dtype=torch.float32) where H > 9 and W > 9
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReflectionPad2d(9)  # Uses padding=9 from the example
        
    def forward(self, x):
        return self.pad(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a valid input tensor with dimensions larger than padding (9)
    return torch.randn(3, 64, 10, 12)  # H=10 >9, W=12 >9 to avoid the documented constraint violation

