import torch.nn as nn

from typing import NamedTuple

import torch


class NT(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


class TSC:
    def __init__(self, nt: NT):
        self.nt = nt


class SomeModule(torch.nn.Module):
    def forward(self, tsc: TSC):
        return tsc.nt.x + tsc.nt.y


if __name__ == '__main__':
    sm = torch.jit.script(SomeModule())
    print(sm(TSC(NT(torch.tensor([1]), torch.tensor([2])))))