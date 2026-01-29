# torch.rand(3, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Dummy forward pass that does not modify the tensor's attributes
        return x + 1  # Example operation to ensure the model is valid

def my_model_function():
    return MyModel()

def GetInput():
    # Create input tensor with an attribute to trigger the deepcopy issue
    x = torch.rand(3, 2)
    x.attr = torch.rand(3, 2)  # Simulate the problematic __dict__ attribute
    return x

