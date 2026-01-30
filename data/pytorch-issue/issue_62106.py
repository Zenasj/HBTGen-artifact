import torch.nn as nn

from torch import fx
from torch import nn

def fx_int(x):
    return int(x)

class MyModule(nn.Module):
    def forward(self, x):
        return fx_int(x.shape[0] / 2)

tracer = fx.Tracer(autowrap_functions=(fx_int,))  # or remove kwarg to demonstrate symbolic trace error
tracer.trace(MyModule())