# torch.rand(3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Reproduces the problematic indexing pattern causing the error
        expected = torch.tensor([x[2, 0], x[1, 1], x[0, 2]])
        return expected

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape (3,4) required by MyModel's forward
    return torch.rand(3, 4, dtype=torch.float32)

