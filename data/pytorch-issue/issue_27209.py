# torch.rand(102, 1, 1, 1, dtype=torch.float32, device="cuda", requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Reshape 4D input to 2D (B, 1) as in original issue
        x_2d = x.view(x.size(0), -1)
        return torch.cdist(x_2d, x_2d, p=2)  # Returns distance matrix that triggers backward error

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces input from original issue with batch size 102
    return torch.randn(102, 1, 1, 1, dtype=torch.float32, device="cuda", requires_grad=True)

