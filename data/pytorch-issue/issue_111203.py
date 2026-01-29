# torch.rand(B, C, H, W, dtype=torch.float32)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 1
        self.register_buffer('index', torch.tensor([0, 1, 2], dtype=torch.long))
        self.register_buffer('source', torch.randn(4, 3, 32, 8))
        self.alpha = 2

    def forward(self, inputs):
        return inputs.index_add(self.dim, self.index, self.source, alpha=self.alpha)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 16, 32, 8)

