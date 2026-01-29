# torch.rand(B, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 1)
        self.b = MyBlock()
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 1)  # Example input with batch size 2

