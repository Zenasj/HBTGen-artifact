# torch.rand(2, 5, 3, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.unfold = nn.Unfold(kernel_size=(3, 3))  # Preserves original configuration
        
    def forward(self, x):
        return self.unfold(x)

def my_model_function():
    return MyModel()  # Direct initialization without extra parameters

def GetInput():
    return torch.randn(2, 5, 3, 4)  # Matches (batch, channels, height, width) from original example

