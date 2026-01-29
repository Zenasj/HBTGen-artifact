# torch.rand(2, 3), torch.rand(2, 3)  # Input shapes for points and query as a tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # Store the dimension as a list of integers

    def forward(self, inputs):
        points, query = inputs  # Unpack the input tuple into points and query
        return (points * query).sum(self.dim)  # Use the stored dim for summing

def my_model_function():
    # Initialize with dim [1] (sum over the second dimension) as a common use case
    return MyModel(dim=[1])

def GetInput():
    B = 2  # Batch size
    C = 3  # Feature channels
    points = torch.rand(B, C)
    query = torch.rand(B, C)
    return (points, query)  # Return a tuple of tensors matching the model's input expectation

