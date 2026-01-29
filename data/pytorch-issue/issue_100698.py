# torch.randint(0, 10, (64, 54, 43), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = [1, -1]
        self.keepdim = True
        # Registered buffer to hold the 'out' tensor with dtype=torch.int32
        self.register_buffer('out', torch.empty(0, dtype=torch.int32))

    def forward(self, x):
        # Construct parameters dictionary as in the original issue's Repro model
        params = {
            'dim': self.dim,
            'keepdim': self.keepdim,
            'input': x,
            'out': self.out
        }
        return torch.sum(**params)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the original issue's shape/dtype
    return torch.randint(0, 10, (64, 54, 43), dtype=torch.int32)

