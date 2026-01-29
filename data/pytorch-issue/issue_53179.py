# torch.rand(1, dtype=torch.float32)  # Input is a dummy scalar
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # First part: replicates the original MyModule's structure
        params1 = [nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
        self.list1 = nn.ParameterList(params1)
        # Second part: replicates the minimal case from the issue
        self.list2 = nn.ParameterList(nn.Parameter(torch.tensor(1.0)) for _ in range(2))
    
    def forward(self, x):
        # Dummy forward pass to satisfy model requirements
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

