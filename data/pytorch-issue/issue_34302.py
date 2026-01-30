import torch
import torch.nn as nn

class Sub(torch.nn.Module):
    def __init__(self, i):
        super(Sub, self).__init__()
        self.i = i
    def forward(self, thing):
        return thing - self.i
    
class M(torch.nn.Module):
    __constants__ = ['mods']
    def __init__(self):
        super(M, self).__init__()
        self.mods = nn.ModuleList([Sub(i) for i in range(10)])
    def forward(self, v):
        v = self.mods[4].forward(v)
        v = self.mods[-1].forward(v)
        v = self.mods[-9].forward(v)
        return v
    
traced_model = torch.jit.script(M())