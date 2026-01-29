# torch.rand(3, 3, 16, 16, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(2, 1), padding=(0,), dilation=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding=(3,), dilation=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.mul(x, x)
        x = self.conv2(x)
        return x

def my_model_function():
    # Return the model instance initialized on CPU
    return MyModel().to('cpu')

def GetInput():
    # Generate input matching the model's expected dimensions
    return torch.randn(3, 3, 16, 16, dtype=torch.float32)

