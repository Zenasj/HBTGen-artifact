# torch.rand(2, dtype=torch.float32)  # Input shape is (2,) as inferred from the provided code

import torch
from torch import nn

class MyModel(nn.Module):
    class Bottleneck(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    def __init__(self):
        super(MyModel, self).__init__()
        self.bottlenecks = nn.ModuleList([self.Bottleneck() for _ in range(3)])

    def forward(self, x):
        y = list([x, x])
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros((2), dtype=torch.float32)

