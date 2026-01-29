# Input is a tuple of 3 tensors each of shape (), e.g., (torch.rand(()), torch.rand(()), torch.rand(()))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)  # Process concatenated tensors

    def forward(self, tensors):
        # tensors is a tuple/list of tensors
        x = torch.dstack(tensors)
        x = x.view(x.size(0), -1)  # Flatten to (batch, features)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tuple of three 0D tensors (scalars)
    return (torch.rand(()), torch.rand(()), torch.rand(()))

