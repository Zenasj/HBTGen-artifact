import torch.nn as nn

import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend="aot_eager", fullgraph=True)
def f(x, i):
    y, z = i.tolist()
    return torch.split(x, [y, z])

print(f(torch.randn(10, requires_grad=True), torch.tensor([7, 3])))

import torch

class Basic(torch.nn.Module):
    def forward(self, x):
        y = x.item()
        return torch.zeros(y + 1)

ep = torch.export.export(Basic(), (torch.tensor(3),))
print(ep)