# torch.rand(1, 64, 40000, dtype=torch.float32)  # Inferred input shape from the issue's example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.trans_conv = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.trans_conv(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected shape and dtype
    return torch.rand(1, 64, 40000, dtype=torch.float32)

