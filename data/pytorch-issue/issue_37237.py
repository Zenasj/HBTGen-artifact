# torch.rand(B, D, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # emb2 initialized as placeholder with assumed shape (5,20) based on index access in original issue
        self.emb2 = nn.Parameter(torch.randn(5, 20)) 

    def forward(self, x):
        # Compute cdist between input and fixed emb2 parameter
        return torch.cdist(x, self.emb2)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input tensor matching emb1's shape (assumed 1x20 based on index [0,17])
    return torch.rand(1, 20, dtype=torch.float32, requires_grad=True)

