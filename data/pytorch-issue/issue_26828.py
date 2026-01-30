import torch.nn as nn

import torch

class ModuleNoTensors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.some_int = 3

    def forward(self, x):
        return x + self.some_int

    def __getstate__(self):
        return (self.some_int,)

    def __setstate__(self, state):
        self.some_int = state[0]

m = ModuleNoTensors()

torch.save(m, 'foo.pkl')
loaded = torch.load('foo.pkl')

print('some int', loaded.some_int)