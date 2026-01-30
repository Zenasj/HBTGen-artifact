import torch.nn as nn

import torch

def fw_hook(inp):
    return tuple([inp[0] + 1])

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(10):
            layer = torch.nn.Linear(16, 16)
            layer.register_forward_pre_hook(lambda _, inp: fw_hook(inp))
            layer = torch.compile(layer, backend='aot_eager')
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

m = Mod()
x = torch.ones(16, 16, requires_grad=True)
out =m(x)