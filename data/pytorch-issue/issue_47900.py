# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicates the scenario where non-scalar output triggers backward error
        self.a = nn.Parameter(torch.ones(2, 2, requires_grad=True))  # Requires grad parameter
    
    def forward(self, x):
        return self.a + x  # Produces non-scalar output (2x2 tensor)

def my_model_function():
    # Returns model instance with initialized parameters
    return MyModel()

def GetInput():
    # Generates 2x2 tensor matching the model's expected input
    return torch.rand(2, 2, dtype=torch.float32)

