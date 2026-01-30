import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def forward(self, x, y):
        return x + y
inputs = (torch.randn(6), torch.randn(6))
export(Foo(), inputs, dynamic_shapes={"x": (DIM.AUTO,), "y": None})

class Foo(torch.nn.Module):
    def forward(self, x, y):
        return x + y
inputs = (torch.randn(6), torch.randn(6))
export(Foo(), inputs, dynamic_shapes={"x": (DIM.AUTO,), "y": {0: Dim("dx", min=3, max=6)}})  # this doesn't work, because x & y and related