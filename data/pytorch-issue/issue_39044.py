# torch.randn(2, 2, 30, 30, dtype=torch.float32, device='cuda')  # Matches the issue's input shape and device
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.FractionalMaxPool2d(3, output_size=(4, 4))  # As per the issue's example configuration

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2, 30, 30, device='cuda')  # Matches the issue's input shape and CUDA device

