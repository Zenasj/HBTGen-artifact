# torch.rand(3, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # Register aa as a buffer to avoid it being treated as an input during export
        self.register_buffer('aa', torch.tensor([[0], [1], [2]], dtype=torch.float32))

    def forward(self, x):
        aa_expanded = self.aa.expand_as(x)
        return self.relu(aa_expanded)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (3,2)
    return torch.rand(3, 2, dtype=torch.float32)

