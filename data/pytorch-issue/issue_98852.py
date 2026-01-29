# torch.rand(2, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.f = nn.Sequential(
            nn.Linear(10, 20),
            self.relu,
            nn.Linear(20, 30),
            self.relu,
            nn.Linear(30, 40),
            self.relu,
        )

    def forward(self, x):
        x = self.f(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 10, dtype=torch.float32)

