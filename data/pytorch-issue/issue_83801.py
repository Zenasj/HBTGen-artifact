import torch.nn as nn

from typing import Dict

import torch


class MyModule(torch.nn.Module):
    tenGrid: Dict[str, torch.Tensor]

    def __init__(self):
        super(MyModule, self).__init__()
        self.tenGrid = {}

    def forward(self, x):
        k = str(x.size())
        if k not in self.tenGrid:
            self.tenGrid[k] = torch.cat((x, x), dim=1)
        return self.tenGrid[k]


torch.onnx.export(torch.jit.script(MyModule()), torch.randn(1, 2), 'test.onnx')