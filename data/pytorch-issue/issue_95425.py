# torch.rand(20, 10, dtype=torch.float32)
import torch
import random
from torch import nn

class MyModule(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(10, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            MyModule(10000, 10000),
            MyModule(10000, 1000),
        )
        self.layers = nn.ModuleList([
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
        ])
        self.tail = nn.Sequential(
            nn.Linear(1000, 5),
        )
        self.layerdrop = 0.5

    def forward(self, x):
        hidden_state = self.pre(x)
        for layer in self.layers:
            dropout_probability = random.uniform(0, 1)
            if dropout_probability < self.layerdrop:
                continue
            hidden_state = layer(hidden_state)
        hidden_state = self.tail(hidden_state)
        return hidden_state

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(20, 10)

