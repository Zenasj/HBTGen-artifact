import torch.nn as nn

import torch
import types

conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True).eval()
class Patch(torch.nn.Module):
    @classmethod
    def patch(cls, self, input, *args, **kwargs):
        return conv(2.0 * input, *args, **kwargs)

m = torch.nn.Module()
m.forward = types.MethodType(Patch.patch, m)

x = torch.randn((1, 3, 32, 32), dtype=torch.float32)
m(x)
exp_program = torch.export.export(m, (x,))