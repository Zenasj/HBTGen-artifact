# torch.rand(B, 500, dtype=torch.float32)  # Input shape (batch, features)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(500, 10)
        
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 500, dtype=torch.float32)

