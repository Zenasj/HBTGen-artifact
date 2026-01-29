# torch.rand(5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.zeros(5, 5, dtype=torch.float32))
        self.register_buffer('indices', torch.tensor([0, 0, 0, 0, 0], dtype=torch.long))

    def forward(self, y):
        return self.x.index_copy(0, self.indices, y)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 5, dtype=torch.float32, requires_grad=True)

