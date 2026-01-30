import torch
import torch.nn as nn
import torch.jit as jit
import torch._jit_internal as _jit

@_jit.weak_module
class MyWeakModule(nn.Module):

    __constants__ = ['linears']

    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])

    @_jit.weak_script_method
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

class MyScriptModule(jit.ScriptModule):

    def __init__(self):
        super().__init__()
        self.linear = MyWeakModule()

    @jit.script_method
    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    model = MyScriptModule()
    x = torch.rand(1, 10)
    out = model(x)
    print(out)