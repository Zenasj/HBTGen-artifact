# torch.rand(26, dtype=torch.float32)  # Input shape inferred from test script's prob tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No trainable parameters; acts as a wrapper for multinomial operation
        self.identity = nn.Identity()  # Placeholder to meet nn.Module requirements
        
    def forward(self, x):
        # Replicates the core operation from the test script
        return torch.multinomial(x, x.numel(), replacement=False)

def my_model_function():
    # Returns the model instance with default settings
    return MyModel()

def GetInput():
    # Reproduces the input from the test script (prob = torch.ones(26))
    return torch.ones(26, dtype=torch.float32)

