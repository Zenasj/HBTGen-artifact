# torch.rand(3, 5, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed tensors from the original issue's test case
        self.register_buffer('a', torch.tensor(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            dtype=torch.float))
        self.register_buffer('index', torch.tensor([0, 4, 2], dtype=torch.long))
    
    def forward(self, x):
        y = x.permute(1, 0)  # Permute to (5, 3) from (3, 5)
        y.index_add_(0, self.index, self.a)  # Modify permuted tensor in-place
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 5, dtype=torch.float)

