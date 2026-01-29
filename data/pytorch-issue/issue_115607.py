# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 2, dtype=torch.float32)

