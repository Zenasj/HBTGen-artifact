# torch.rand(1, 1000, 1000, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.cuda(0)  # Explicitly move to GPU 0

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1000, 1000, 100, dtype=torch.float32)

