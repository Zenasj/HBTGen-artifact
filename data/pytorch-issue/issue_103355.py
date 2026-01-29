# torch.rand(5, 7, 3, dtype=torch.float32)  # Inferred input shape from test_bool_indices
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Boolean mask used in test_bool_indices (shape (5,))
        self.bool_mask = torch.tensor([True, False, True, True, False], dtype=torch.bool)
    
    def forward(self, x):
        # Perform boolean indexing on first dimension
        return x[self.bool_mask]

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching test's (5,7,3) shape
    return torch.rand(5, 7, 3, dtype=torch.float32)

