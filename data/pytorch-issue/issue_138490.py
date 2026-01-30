import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def forward(self, x, y):
        return x - y

inputs = (torch.randn(4, 4), torch.randn(4, 4))
shapes = {
    "x": (Dim.AUTO, Dim("d1", min=3)),
    "y": (Dim("d0", max=8), Dim.DYNAMIC),
}
ep = export(Foo(), inputs, dynamic_shapes=shapes)