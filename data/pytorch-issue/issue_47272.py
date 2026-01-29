# torch.rand(2, 3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute batched inverse and split-inverse comparison
        batch_inv = torch.inverse(x)
        split_inv = torch.stack([torch.inverse(m) for m in x])
        # Return boolean tensor indicating if results match (within tolerance)
        return torch.tensor(torch.allclose(batch_inv, split_inv, atol=1e-6), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random 2x3x3 tensor (matching the issue's problematic batch size)
    return torch.rand(2, 3, 3, dtype=torch.float32)

