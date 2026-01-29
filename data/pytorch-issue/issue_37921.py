# torch.rand(B, N, dtype=torch.float32)
import torch
from torch import nn

class CPUBehaviorMax(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val, _ = torch.max(x, dim=self.dim)
        reversed_x = torch.flip(x, [self.dim])
        _, reversed_indices = torch.max(reversed_x, dim=self.dim)
        last_indices = (x.size(self.dim) - 1 - reversed_indices)
        return max_val, last_indices

class CUDABehaviorMax(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        values, _ = torch.max(x, dim=self.dim)
        indices = torch.argmax(x, dim=self.dim)
        return values, indices

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpu_behavior = CPUBehaviorMax(dim=1)
        self.cuda_behavior = CUDABehaviorMax(dim=1)
    
    def forward(self, x):
        _, cpu_indices = self.cpu_behavior(x)
        _, cuda_indices = self.cuda_behavior(x)
        return (cpu_indices == cuda_indices).all()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, dtype=torch.float32)

