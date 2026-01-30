import torch.nn as nn

import torch
from torch.export import export

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.tensor(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.a

def forward_hook(
    module: torch.nn.Module, inputs, output
) -> torch.Tensor:
    return 2 * output

seq = torch.nn.Sequential(TestModule()).eval()
seq.b = torch.tensor(2)
handle = seq.register_forward_hook(forward_hook)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = seq

    def forward(self, x):
        return self.seq(x) + self.seq.b

inp = (torch.randn(2, 8),)
ep = export(M(), inp)  # This errors because dynamo adds an extra input