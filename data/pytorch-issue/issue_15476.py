# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size (can be any positive integer)
    return torch.rand(B, 3, dtype=torch.float32)

