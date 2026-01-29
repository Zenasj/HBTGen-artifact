# torch.rand(B=1, C=1, H=16, W=16, dtype=torch.float32)  # Shape from original repro code
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  # Matches SimpleConv structure from issue

    def forward(self, x):
        return self.conv1(x)

def my_model_function():
    # Returns initialized MyModel instance
    return MyModel()

def GetInput():
    # Returns random input matching (N,C,H,W) = (1,1,16,16)
    return torch.rand(1, 1, 16, 16, dtype=torch.float32)

