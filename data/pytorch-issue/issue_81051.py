# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('row_indices', torch.tensor([1, 2], dtype=torch.long))
        self.register_buffer('col_indices', torch.tensor([2, 1], dtype=torch.long))
    
    def forward(self, x):
        return x[self.row_indices, self.col_indices]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)

