import torch
import torch.nn as nn


class M(torch.jit.ScriptModule):

    def __init__(self):
        super(M, self).__init__()
        self.softmax = nn.Softmax(dim=0)

    @torch.jit.script_method
    def forward(self, v):
        return self.softmax(v)

i = torch.Tensor(2)
m = M()
o = m(i)

import torch.nn.functional as F


@torch.jit.script
def test(input, label):
    return F.cross_entropy(input, label)


x = torch.randn(5, 5)
y = torch.randint(size=(5,), high=5, dtype=torch.long)
print(test(x, y))