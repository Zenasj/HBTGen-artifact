# torch.rand(10, 10, 5, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = -1
        self.sparse_grad = True
        # The index tensor as per the issue's example (shape [6, 0, 2], dtype=torch.uint8)
        self.register_buffer('index', torch.randint(0, 1, (6, 0, 2), dtype=torch.uint8))

    def forward(self, x):
        return torch.gather(x, self.dim, self.index, sparse_grad=self.sparse_grad)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, 5, dtype=torch.float64)

