# torch.randint(0, 10, (14,), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.arange(10.0))  # dtype=torch.float32

    def forward(self, indices):
        # Clone to avoid in-place modification of parameters
        output = self.x.clone()
        output.scatter_(0, indices, 0.0)
        return output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (14,), dtype=torch.int64)

