# torch.randint(low=0, high=10, size=(), dtype=torch.int64)  # Input is a scalar tensor for 'high' parameter
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.size = (3960,)  # Fixed size from the original test case
        
    def forward(self, high):
        return torch.randint(high=high, size=self.size)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random scalar integer tensor as input to MyModel
    return torch.randint(low=0, high=10, size=(), dtype=torch.int64)

