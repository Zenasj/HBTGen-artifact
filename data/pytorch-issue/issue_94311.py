import torch.nn as nn

import torch


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zero = torch.tensor(0)

    def forward(self):
        return self.zero.round().int()

dummy = Dummy()
traced = torch.jit.script(dummy)
traced()
traced()  # the bug only appears on the second time the module is run