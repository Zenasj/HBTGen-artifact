# torch.rand(2, 3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.num_batches_tracked = 0  # Problematic attribute modified in forward

    def forward(self, x):
        self.num_batches_tracked += 1  # Triggers ONNX export error when scripted
        return x

def my_model_function():
    return MyModel()  # Returns the problematic model instance

def GetInput():
    return torch.rand(2, 3, dtype=torch.float)  # Matches input shape (B=2, C=3)

