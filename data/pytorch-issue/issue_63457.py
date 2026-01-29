# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.param = nn.Parameter(torch.randn(2, 2))  # Matches the original model's parameter
        
    def forward(self, x):
        # Example operation using the parameter (for compatibility with torch.compile)
        return self.param * x  # Element-wise multiplication as a dummy forward pass

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (2, 2) as required by MyModel's forward
    return torch.rand(2, 2, dtype=torch.float32)

