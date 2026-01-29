# torch.rand(1, dtype=torch.int32)  # Input is a scalar tensor indicating dataset size
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_samples=None, replacement=False):
        super().__init__()
        self.num_samples = num_samples
        self.replacement = replacement

    def forward(self, n):
        n = n.item()  # Extract scalar value from tensor input
        if not self.replacement and self.num_samples is not None:
            indices = torch.randperm(n)[:self.num_samples]
        else:
            indices = torch.randperm(n)
        return indices

def my_model_function():
    # Example initialization with proposed num_samples and replacement=False
    return MyModel(num_samples=5, replacement=False)

def GetInput():
    # Return dataset size as scalar tensor (e.g., 10 elements)
    return torch.tensor([10], dtype=torch.int32)

