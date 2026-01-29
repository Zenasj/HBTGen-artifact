# torch.rand(B, 20, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # New weight norm implementation (fix from parametrizations)
        self.layer = weight_norm(nn.Linear(20, 40), name='weight')

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Random input tensor with shape (batch, 20)
    return torch.rand(1, 20)  # B=1 for simplicity

