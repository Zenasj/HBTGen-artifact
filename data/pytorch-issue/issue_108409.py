# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 3),
            nn.Linear(3, 3),
            nn.Linear(3, 3)
        )
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, requires_grad=True)

