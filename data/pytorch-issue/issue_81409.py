# torch.rand(1, 1, 5877, 3697, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, output_size=(2855, 1)):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x):
        return torch._C._nn.adaptive_avg_pool2d(x, self.output_size)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 5877, 3697, dtype=torch.float32)

