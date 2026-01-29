# torch.rand(1, 2, 4, 4, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(2, 3, 3)  # Matches the original Conv2d configuration
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input tensor matching the expected shape and requirements
    return torch.rand(1, 2, 4, 4, dtype=torch.float32, requires_grad=True)

