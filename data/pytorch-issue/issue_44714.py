# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn
from torch.utils.data import random_split

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.Generator(device='cuda:0')  # CUDA generator causing the issue

    def forward(self, x):
        # Reproduce the bug by calling random_split with a CUDA generator on CPU data
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # CPU-based data
        try:
            splits = random_split(data, [8, 2], generator=self.generator)
            return x  # Return input if no error (but the split will fail)
        except Exception as e:
            return torch.tensor([0])  # Dummy output to avoid crashing the forward pass

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy 4D tensor to match expected input shape (arbitrary dimensions)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

