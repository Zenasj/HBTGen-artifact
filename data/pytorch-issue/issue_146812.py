# torch.rand(B, C, H, dtype=torch.float32)  # Inferred from test input shape (11, 15, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Directly use the aten operator mentioned in the test case
        return torch.ops.aten._pdist_forward(x, p=2.0)

def my_model_function():
    # Returns instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape from the test case
    return torch.rand(11, 15, 3, dtype=torch.float32)

