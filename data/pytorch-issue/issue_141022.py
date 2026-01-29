# torch.rand(2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, a):
        b = a + 1
        c = b.view(-1)
        c.add_(1)  # In-place operation as in original code
        return b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2)

