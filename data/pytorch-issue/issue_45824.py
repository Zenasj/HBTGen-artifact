# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Example batch size
    return torch.rand(B, 10, dtype=torch.float32)

