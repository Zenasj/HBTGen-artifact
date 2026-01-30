import torch.nn as nn

import torch as th

class Mod(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return th.cat(2*[x], dim=0)
        #return th.cat((x, x), dim=0) <-- Unlike before, this will still cause the load below to fail.

class ScriptMod(th.jit.ScriptModule):
    def __init__(self, mod):
        super().__init__()
        x = th.zeros(1, 3)
        mod_fn = lambda : mod(x)
        self.mod = th.jit.trace(mod_fn, tuple())

    @th.jit.script_method
    def forward(self):
        return self.mod()

if __name__ == "__main__":
    with th.no_grad():
        cm = ScriptMod(Mod())
        cm.save("mod.ptj")
        cm = th.jit.load("mod.ptj") # <-- This will fail with the same error as in the original repro.