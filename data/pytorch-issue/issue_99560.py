# torch.rand(B, C, dtype=torch.float32)  # Input is 2D tensor for transpose operation
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Simulate the transposed operation (equivalent to NumPy's .T)
        return x.transpose(0, 1)  # Transpose dimensions for 2D input

def my_model_function():
    return MyModel()

def GetInput():
    # Generate 2D input tensor matching the example's use of .T
    B, C = 2, 3  # Example batch and channel dimensions
    return torch.rand(B, C, dtype=torch.float32)

