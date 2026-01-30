import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, x):
        return x.sin()

def f(m):
    return callable(m)

res = torch.compile(f, fullgraph=True)(M())