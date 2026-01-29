# torch.rand(B, 30, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(30, 10)
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size
    return torch.rand(B, 30, dtype=torch.float32)

