# torch.rand(1, 2, 1, 1, dtype=torch.float32)  # Inferred input shape (B=1, C=2, H=1, W=1)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Minimal model to reproduce CUDA initialization issue
        # Forward pass simply returns input to trigger CUDA context
        pass

    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random 4D tensor matching the inferred input shape on CUDA
    return torch.rand(1, 2, 1, 1, dtype=torch.float32, device=torch.device('cuda'))

