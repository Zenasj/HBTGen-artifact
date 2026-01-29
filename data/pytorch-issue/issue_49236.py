# torch.rand(2, 2, dtype=torch.float, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Store intermediate tensor to allow in-place detachment later
        self.b = 2 * x  
        c = self.b.sum()
        return c

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float, requires_grad=True)

