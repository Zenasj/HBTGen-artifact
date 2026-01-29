# torch.rand(B, 2, 3, dtype=torch.float32)  # Inferred input shape from issue example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Apply nansum and nanmean along dimension 1 as per issue context
        sum_result = torch.nansum(x, dim=1)
        mean_result = torch.nanmean(x, dim=1)
        # Concatenate outputs for demonstration purposes
        return torch.cat([sum_result, mean_result], dim=1)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (B, 2, 3)
    B = 32  # Batch size from issue example
    return torch.rand(B, 2, 3, dtype=torch.float32)

