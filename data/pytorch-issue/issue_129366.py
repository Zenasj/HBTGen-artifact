# torch.rand(1, dtype=torch.float32)  # Dummy input; model uses stored nested tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create nested tensor with layout=torch.jagged to trigger serialization issue
        g0 = torch.zeros(1)
        g1 = torch.zeros(2)
        self.nt = torch.nested.nested_tensor([g0, g1], layout=torch.jagged)
        # Access .shape to cache the problematic PyCapsule (as per repro steps)
        self.nt.shape

    def forward(self, x):
        # Forward pass returns the nested tensor (x is unused but required for input compatibility)
        return self.nt

def my_model_function():
    return MyModel()

def GetInput():
    # Return dummy input compatible with MyModel's forward (any tensor works here)
    return torch.rand(1)

