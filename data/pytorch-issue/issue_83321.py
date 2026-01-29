# torch.rand(B, C, D, H, W, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Kernel size (3,2,2) and stride=1 as per issue's parameters
        self.pool = nn.AvgPool3d(kernel_size=(3, 2, 2), stride=1)
    
    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape [20,16,50,44,31] and bfloat16 dtype from the issue
    return torch.rand(20, 16, 50, 44, 31, dtype=torch.bfloat16)

