import numpy as np

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = torch.rand(1,2,1,1)
        self.layer = nn.Conv2d(
            2, 1, kernel_size=(1,2),
        )

    def forward(self, i0):
        x = torch.multiply(i0, self.const)
        x = self.layer(x)
        o0 = torch.clip(x, -1.5, 1.5)
        o1 = torch.floor(x)
        return o0, o1

def build():
    inp = {
        'i0': torch.rand(1,2,2,2,),
    }

    mod = MyModule()
    exported = torch.jit.trace(mod, list(inp.values()))
    exported = torch.jit.optimize_for_inference(exported)

    out = mod(**inp)
    print(f'{out = }')

    eout = exported(**inp)
    print(f'{eout = }')

    assert torch.allclose(out[0], eout[0])

build()