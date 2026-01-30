import torch
import torch.nn as nn

class MultiBox(torch.jit.ScriptModule):

    __constants__ = ['loc_layers']

    def __init__(self):
        super(MultiBox, self).__init__()

        self.loc_layers = nn.ModuleList()
        for i in range(4):
            self.loc_layers.append(nn.Conv2d(64, 4, kernel_size=1))

    @torch.jit.script_method
    def forward(self, x):
        return x

class MultiBox2(torch.jit.ScriptModule):

    __constants__ = ['loc_layers']

    def __init__(self):
        super(MultiBox2, self).__init__()

        loc_layers = nn.ModuleList()
        for i in range(4):
            loc_layers.append(nn.Conv2d(64, 4, kernel_size=1))

        self.loc_layers = loc_layers

    @torch.jit.script_method
    def forward(self, x):
        return x