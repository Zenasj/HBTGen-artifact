# torch.rand(1, 16, 100, 100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=3, padding=1)

    def forward(self, x):
        n, c, h, w = x.shape
        unfold_x = self.unfold(x).view(n, -1, h, w)
        return unfold_x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 16, 100, 100, dtype=torch.float32)

