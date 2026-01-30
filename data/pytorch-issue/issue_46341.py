import torch
import torch.nn as nn

@torch.jit.script
class ValHolder(object):
    def __init__(self, val):
        self.val = val

class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.mod1 = ValHolder(1)
        self.mod2 = ValHolder(2)

    def forward(self, cond: bool):
        if cond:
            mod = self.mod1
        else:
            mod = self.mod2
        return mod.val

mod = Mod()
print(torch.jit.script(mod)())