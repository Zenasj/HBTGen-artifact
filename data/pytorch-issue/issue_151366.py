import torch.nn as nn

import torch
from torch.testing._internal.custom_tensor import CustomTensorPlainOut

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.ones(3, 4))
        self.p2 = torch.nn.Parameter(
            CustomTensorPlainOut(
                torch.ones(3, 4),
                torch.ones(3, 4),
            )
        )

    def forward(self, x):
        a = (2 * self.p1 + self.p2).sum()
        return x + a

model = Foo()
example_inputs = (torch.randn(3, 4),)
ep = torch.export.export(model, example_inputs, strict=False)
ep.run_decompositions()
ep.module()