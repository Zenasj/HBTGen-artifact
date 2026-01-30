import torch.nn as nn

from fractions import gcd

import torch
from torch import nn


class CrossScale(torch.jit.ScriptModule):
    __constants__ = ['xc']

    def __init__(self, n, ng=32):
        super(CrossScale, self).__init__()

        xc = []
        for i in range(len(n)):
            m = nn.Sequential(
                nn.Conv2d(n[i], n[i], 1, bias=False),
                nn.GroupNorm(gcd(ng, n[i]), n[i]))
            xc.append(m)
        self.xc = nn.ModuleList(xc)

    @torch.jit.script_method
    def forward(self, x):
        # type: (List[Tensor]) -> List[Tensor]
        outputs = []
        for cur_xc in self.xc:
            out = cur_xc(torch.tensor(0))
            outputs.append(out)
        return outputs


if __name__ == "__main__":
    n = [32, 64, 96]
    cs = CrossScale(n)

    cs.save("cs.pt")