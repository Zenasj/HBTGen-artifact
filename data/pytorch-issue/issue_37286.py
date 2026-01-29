# torch.rand(B, 6, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Replicate original logic with TorchScript-compatible indexing
        transposed = x.t()
        indices = torch.tensor([0, 1, 5], device=x.device)
        return transposed.index_select(0, indices)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 6, dtype=torch.float)

