import torch.nn as nn

import torch as th

class TestMod(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = th.nn.ParameterList([th.nn.Parameter(th.zeros(3)), th.nn.Parameter(th.zeros(3))])

    def forward(self, x):
        return x + self.params[0] + self.params[1]

mod = TestMod()
smod = th.jit.script(mod)