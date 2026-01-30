import torch.nn as nn

import torch
from torch.jit import Final
print(torch.__version__)

class Model(torch.nn.Module):
    a: Final[int]

    def __init__(self):
        super().__init__()
        self.a = 5

    def forward(self, x):
        return x

m = torch.jit.script(Model())
torch.jit.save(m, 'model.pt')
torch.jit.load('model.pt')

import torch
print(torch.__version__)

m = torch.jit.script(torch.nn.Linear(10, 10))
torch.jit.save(m, 'model.pt')
torch.jit.load('model.pt')