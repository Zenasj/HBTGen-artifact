# torch.rand(B, 200, dtype=torch.float32)

import torch
import torch.nn as nn
from torch.nn import Module, ModuleList

class Sequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, input):
        for module in self.children():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

class Parallel(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, input):
        outputs = []
        for module in self.children():
            outputs.append(module(input))
        return tuple(outputs)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            nn.Linear(200, 100),
            Parallel(
                nn.Linear(100, 16),
                nn.Linear(100, 16),
            ),
            nn.Bilinear(16, 16, 100),
        )

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 200, dtype=torch.float32)

