# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Identity module to match the issue's tensor shape context
        self.identity = nn.Identity()
    
    def forward(self, x):
        return self.identity(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the shape discussed in the issue (40,100,256,256)
    return torch.rand(40, 100, 256, 256, dtype=torch.float32)

