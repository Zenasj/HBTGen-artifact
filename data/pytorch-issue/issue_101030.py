import torch._dynamo
import torch
import torch.nn as nn

class Model(nn.Module):
    export = False

    def __init__(self, linear):
        super().__init__()
        self.m = nn.ModuleList([linear] * 3)

    def forward(self, x):
        for i, mod in enumerate(self.m):
            if not self.training:
                x = mod(x)
        return x


@torch._dynamo.optimize("eager")
def f(l, x):
    layers = []
    layers += [Model(l)]
    m = nn.Sequential(*layers)
    return m(x)

linear = nn.Linear(3,3)
x = torch.randn(1, 3)
t = f(linear, x)