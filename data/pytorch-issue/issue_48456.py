# torch.rand(B, N, N), torch.rand(B, N)  # Input shapes for A and B as a tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        A, B = inputs
        return torch.linalg.solve(A, B)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    N = 3  # Matrix dimension (NxN)
    A = torch.rand(B, N, N, dtype=torch.float32)  # Batch of square matrices
    B_tensor = torch.rand(B, N, dtype=torch.float32)  # Batch of right-hand sides (1D per batch)
    return (A, B_tensor)  # Tuple matches model's expected input format

