import torch
import torch.nn as nn
import torch.nn.functional as F

class M1(torch.nn.Module):
    def forward(self, x):
        return F.relu(x)

m = M1()

m = decompose(m)
m.graph.lint()
print(m.code)

def forward(self, x):
    gt = x > 0
    mul = gt * x;  gt = x = None
    return mul