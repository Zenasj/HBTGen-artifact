import torch
import torch.nn as nn

# torch.rand(3, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = MinimalExampleOriginal()
        self.fixed = FixedExample()
    
    def forward(self, x):
        _, corr_orig = self.original(x)
        _, corr_fixed = self.fixed(x)
        return torch.stack([corr_orig, corr_fixed])

class MinimalExampleOriginal(nn.Module):
    def forward(self, x):
        perm = torch.randperm(x.numel())
        correct = torch.tensor(perm.dtype == torch.int64, dtype=torch.bool)
        return x[perm], correct

class FixedExample(nn.Module):
    def forward(self, x):
        perm = torch.randperm(x.numel(), dtype=torch.int64)
        correct = torch.tensor(perm.dtype == torch.int64, dtype=torch.bool)
        return x[perm], correct

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3)

