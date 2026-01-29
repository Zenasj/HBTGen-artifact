# torch.rand(2, dtype=torch.int64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x  # Identity model to pass through DataLoader output

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random integer tensor matching DataLoader batch shape
    return torch.randint(0, 20, (2,), dtype=torch.int64)

