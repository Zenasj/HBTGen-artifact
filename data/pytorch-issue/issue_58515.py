import torch.nn as nn

import torch

class Module(torch.nn.Module):

    def __init__(self, strings):
        super().__init__()
        self.inner = Inner(strings)
        self.map = {k: j for j, k in enumerate(strings)}

    def forward(self, x):
        return x


class Inner:

    def __init__(self, strings):
        self.map = {k: j for j, k in enumerate(strings)}

m = torch.jit.script(Module(['A', 'B', 'C']))

class Inner:
    def __init__(self, strings: List[str]):
        self.map: Dict[str, int] = {k: j for j, k in enumerate(strings)}