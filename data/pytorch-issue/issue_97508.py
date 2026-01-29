# torch.rand(4, 4)  # Inferred input shape: 2D tensor of size (4,4)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_tensor = torch.tensor([2, 8], dtype=torch.int64)  # Example shape parameter as a tensor

    def forward(self, x):
        # Dynamically reshape using tensor-based shape parameter
        return x.reshape(self.shape_tensor.tolist())  # Convert tensor to list for reshape compatibility in non-Dynamo mode

def my_model_function():
    # Returns the model instance with the tensor-based shape parameter
    return MyModel()

def GetInput():
    # Returns a random 4x4 input tensor matching the model's expected input
    return torch.rand(4, 4)

