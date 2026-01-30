import torch
import torch.nn as nn

class M(nn.Module):
    __constants__ = ['my_const']
    def __init__(self):
        super().__init__()
        self.my_const = 2

    def forward(self, x):
        return x + self.my_const


sm = torch.jit.script(M())

# AttributeError: 'RecursiveScriptModule' object has no attribute 'my_const'
print(sm.my_const)