# torch.rand(2, 4, dtype=torch.float64)  # Inferred input shape from constant tensor dimensions and storage type
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Constant tensor from the issue's constants.pkl (shape (2,4), DoubleStorage)
        self.constant = nn.Parameter(torch.rand(2, 4, dtype=torch.float64), requires_grad=False)
        # Scalar constant from the issue's constants.pkl (value 1)
        self.scalar = 1  # Stored as a module attribute for forward computation

    def forward(self, x):
        # First add: constant tensor + input x (assumed to be a tensor of same shape)
        # Second add: result + scalar (1)
        return (self.constant + x) + self.scalar

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the constant tensor's shape and dtype
    return torch.rand(2, 4, dtype=torch.float64)

