import torch.nn as nn

import torch
import torch._dynamo.config

class M(torch.nn.Module):
    step_count = 0

    def forward(self, x):
        return x * self.step_count

m = M()

@torch.compile(fullgraph=True)
def f(a):
    return m(a)

f(torch.randn(3))
m.step_count += 1
f(torch.randn(3))
m.step_count += 1
f(torch.randn(3))