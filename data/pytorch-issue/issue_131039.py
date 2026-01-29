# torch.randn(B, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return (x + 1).pow(1 / 13) - 1

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size can be arbitrary (set to 1 for minimal reproduction)
    return torch.randn(B, 5, dtype=torch.float32)

