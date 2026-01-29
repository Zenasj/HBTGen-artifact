# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example parameter for comparison (similar to y_tensor in benchmarks)
        self.y_tensor = nn.Parameter(torch.ones(2, 1, dtype=torch.float32))

    def forward(self, x):
        # Perform comparisons with scalar, 1-element tensor, and broadcasted tensor
        # Returns a tuple of comparison results for testing purposes
        return (
            x == 1.0,          # Scalar comparison
            x == torch.tensor(1.0),  # 0D tensor comparison
            x == self.y_tensor # Broadcasted tensor comparison (shape (2,2) vs (2,1))
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Input tensor matching the shape used in the PR's benchmarks
    return torch.rand(2, 2, dtype=torch.float32)

