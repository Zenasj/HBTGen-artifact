# torch.rand(1, 10, 32, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # This model demonstrates usage of PyStructSequence return types
        # (e.g., from torch.topk) which require _fields attribute handling

    def forward(self, x):
        # Returns a PyStructSequence (namedtuple-like) from torch.topk
        return torch.topk(x, k=5, dim=1)

def my_model_function():
    # Returns a model instance that outputs PyStructSequence types
    return MyModel()

def GetInput():
    # Returns random tensor matching expected input shape
    return torch.randn(1, 10, 32, 32, dtype=torch.float32)

