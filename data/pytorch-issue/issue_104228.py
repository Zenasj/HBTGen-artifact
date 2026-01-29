# torch.randint(0, 10, (1, 3, 224, 224), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x << 0  # Triggers SymInt bitshift graph break in Dynamo

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1, 3, 224, 224), dtype=torch.int32)

