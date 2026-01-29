# torch.rand(1, 3, 224, 224, dtype=torch.float), torch.rand(1, 3, 224, 224, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.tensor(3.0))  # Matches the integer input from the example
        
    def forward(self, inputs):
        t0, t1 = inputs
        return t0 + t1 + self.x  # Replicates the logic of the first example's "foo" function

def my_model_function():
    return MyModel()

def GetInput():
    # Returns two tensors matching the expected tuple input
    return (torch.rand(1, 3, 224, 224, dtype=torch.float),
            torch.rand(1, 3, 224, 224, dtype=torch.float))

