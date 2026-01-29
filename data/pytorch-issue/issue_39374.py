# torch.rand(1, 32, 24, 512, 512, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        indices = x < 0
        x[indices] = 0.0  # Reproduces memory-intensive indexing operation
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 32, 24, 512, 512, dtype=torch.float32)

