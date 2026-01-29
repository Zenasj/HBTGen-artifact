# torch.rand(1, 2, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy module to mirror RPC test scenario
        self.add_op = nn.Identity()  # Matches torch.add behavior in test
        
    def forward(self, x):
        # Simulate RPC-encapsulated operation (e.g., torch.add)
        return self.add_op(x + 2)  # Matches test's torch.add(ones(2), 2)

def my_model_function():
    # Returns model instance with minimal initialization
    return MyModel()

def GetInput():
    # Matches test's input dimensions (adjusted to 4D tensor)
    return torch.rand(1, 2, 1, 1, dtype=torch.float32)

