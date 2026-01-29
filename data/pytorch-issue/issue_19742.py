# torch.rand(1, dtype=torch.float32)  # Inferred input shape (scalar tensor)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Creates new tensors each call leading to non-deterministic pickling
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([1.0], dtype=torch.float32)
        return a + b

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a scalar tensor to trigger the non-deterministic pickle issue
    return torch.rand(1, dtype=torch.float32)

