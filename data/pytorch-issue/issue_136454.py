# torch.rand(B, C, H, dtype=torch.float32)
import torch
from torch import nn, linalg

class MyModel(nn.Module):
    def forward(self, x):
        has_mut = False
        has_must = False
        try:
            linalg.norm(x, ord='fro', dim=(0, 1, 2))
        except RuntimeError as e:
            has_mut = 'mut' in str(e)
        try:
            linalg.norm(x, ord=2, dim=(0, 1, 2))
        except RuntimeError as e:
            has_must = 'must' in str(e)
        return torch.tensor([1.0 if has_mut and has_must else 0.0])

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 4, dtype=torch.float32)

