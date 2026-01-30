import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self):
        full = torch.full((), 11)
        i0 = full.item()
        return (torch.full((i0,), 0), )

args = ()
gm = torch.export.export(M(), ())  # .module()