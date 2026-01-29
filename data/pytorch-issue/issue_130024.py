# torch.rand(2, 2, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Indices and source tensors defining overlapping scatter writes
        self.register_buffer('indices', torch.tensor([[0, 0], [0, 1]], dtype=torch.long))
        self.register_buffer('src', torch.tensor([[1, 2], [3, 4]], dtype=torch.float))
        self.dim = 0  # Scatter along rows (dimension 0)

    def forward(self, x):
        # Clone input to avoid in-place modification
        x_scatter = x.clone()
        x_scatter.scatter_(self.dim, self.indices, self.src)
        return x_scatter

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float)

