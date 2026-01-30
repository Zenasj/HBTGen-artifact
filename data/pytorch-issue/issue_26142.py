import torch.nn as nn

import torch

class Bug(torch.nn.Module):
    def forward(self, x):
        # no bug
#       bug: "IndentationError: unexpected indent"
        return x

f = torch.jit.script(Bug())