import torch.nn as nn

import torch

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.zeros([]).dtype

    def forward(self):
        return torch.zeros(3, 4, dtype=self.dtype)


f = Foo()
torch.jit.script(f)