import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def forward(self, x, y, z):
        ...

    foo = Foo()
    inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))

for dynamic_shapes in [
    None
    ((4, 6), (5, 4), (3, 3)),
    ((None, 6), None, {0: 3, 1: 3})
]:
    _ = export(foo, inputs, dynamic_shapes=dynamic_shapes)

_ = export(foo, inputs, dynamic_shapes=((5, None), None, None))