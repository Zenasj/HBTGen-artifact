import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def forward(self, x, y):
        return x.reshape([-1]) + y

dy = Dim("dy", min=6)
x, y = torch.randn(6, 2), torch.randn(12)
dynamic_shapes = {
    "x": (dy - 6, 2),
    "y": (dy,),
}