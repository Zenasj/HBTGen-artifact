# torch.rand(B, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1024, 64, dtype=torch.float32)

