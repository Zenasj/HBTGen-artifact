# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class NetA(nn.Module):
    @staticmethod
    def splitA(x):
        return torch.split(x, 3)
    
    def forward(self, x):
        return self.splitA(x)

class NetB(nn.Module):
    @staticmethod
    def splitB(x, split_size: int):
        return torch.split(x, split_size)
    
    def forward(self, x):
        return self.splitB(x, 3)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA = NetA()
        self.netB = NetB()
    
    def forward(self, x):
        outA = self.netA(x)
        outB = self.netB(x)
        # Check if all tensors in output tuples are equal
        if len(outA) != len(outB):
            return torch.tensor(False)
        for a, b in zip(outA, outB):
            if not torch.allclose(a, b):
                return torch.tensor(False)
        return torch.tensor(True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 6, 32, 32, dtype=torch.float32)

