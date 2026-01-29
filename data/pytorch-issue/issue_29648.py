# torch.rand(B=1, C=1, H=16, W=16, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.quantization import MinMaxObserver

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.observer = MinMaxObserver()  # Fixed observer as per the issue resolution

    def forward(self, x):
        # Forward pass applies the observer to collect statistics
        return self.observer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 16, 16, dtype=torch.float32)

