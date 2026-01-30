import torch.nn as nn

import torch

print(torch.__version__)  # Tried with 2.2.0+cu121 (latest)


class Scalar(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        return self.scalar * x


class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        scalar = Scalar(2.0)
        x = super().forward(*args, **kwargs)
        return scalar(x)


m = Linear(4, 8)
m(torch.randn(3, 4))  # passes

compiled_m = torch.compile(m)  # passes
compiled_m(torch.randn(3, 4))  # fails