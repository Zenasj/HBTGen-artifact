# torch.rand(8, 4, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.glu = nn.GLU(dim=1)  # Split along channel dimension (C)

    def forward(self, x):
        return self.glu(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 4, 1, 1, dtype=torch.float32)

