import torch.nn as nn

import torch

class Foo(torch.nn.Module):
    pass

class Abc(torch.nn.Module):
    def forward(self):
        # If you comment the following line, compilation fails because it can't resolve Foo
        Foo.__class__
        if isinstance(self, Foo):
            return torch.zeros(1)
        else:
            return torch.ones(1)

scr = torch.jit.script(Abc())