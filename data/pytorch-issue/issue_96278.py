# torch.rand(1, dtype=torch.float32)  # Dummy input to satisfy model signature
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the error scenario of creating a tensor inside compiled forward
        return torch.tensor(data=[[1., -1.]])

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input required for model compatibility
    return torch.rand(1, dtype=torch.float32)

