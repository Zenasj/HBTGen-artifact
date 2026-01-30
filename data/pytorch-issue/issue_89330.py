import torch
import torch.nn as nn

class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super(MyModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            *[MyModule(10, 10000)]
            + [MyModule(10000, 1000)]
            + [MyModule(1000, 5)]
        )

    def forward(self, x):
        return self.net(x)