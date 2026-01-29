# torch.rand(5, dtype=torch.float32, device='cuda')

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('indices', torch.tensor([1, 3], dtype=torch.long))

    def forward(self, x):
        # Operation that triggers the indexing error under deterministic mode
        x[self.indices] = 2  # Assign scalar to multiple indices
        return x

def my_model_function():
    return MyModel().cuda()

def GetInput():
    return torch.rand(5, dtype=torch.float32, device='cuda')

