# torch.rand(1, 1, 5, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stride1 = nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=1)
        self.conv_stride2 = nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2)
    
    def forward(self, x):
        # Returns outputs of both models as a tuple
        return self.conv_stride1(x), self.conv_stride2(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Valid input for both convolutions (size 5x5 to avoid zero-output for stride=1)
    return torch.rand(1, 1, 5, 5, dtype=torch.float32)

