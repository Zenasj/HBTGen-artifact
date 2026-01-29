# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(10, 20)
        self.g2 = nn.Linear(20, 30)
        self.g3 = nn.Linear(30, 40)

    def forward(self, x):
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size can be adjusted as needed
    return torch.rand(B, 10, dtype=torch.float32)

