# torch.nested.nested_tensor([torch.rand(1, 3), torch.rand(2, 2)], dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Extract first two elements of the nested tensor for identity comparison
        a = x[0]
        b = x[1]
        return (a, b)  # Return both tensors to enable external ID checks

def my_model_function():
    return MyModel()

def GetInput():
    # Create nested tensor with example tensors from the issue
    ts = [
        torch.tensor([[1.0, 2.0, 3.0]]),  # First tensor (shape 1x3)
        torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Second tensor (shape 2x2)
    ]
    return torch.nested.nested_tensor(ts)  # Matches nested tensor input structure

