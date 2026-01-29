# torch.rand(1, 16, 70000, 100, dtype=torch.float32).to('cuda:0')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(16, 32, kernel_size=3, bias=False)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Initialize model on CUDA to reproduce the error scenario
    return MyModel().to('cuda:0')

def GetInput():
    # Generate input matching the failing case dimensions
    return torch.rand(1, 16, 70000, 100, dtype=torch.float32).to('cuda:0')

