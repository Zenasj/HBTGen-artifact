import torch.nn as nn

import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = torch.tensor([True, False])
    def forward(self, x):
        x.view(3, 2).masked_fill_(self.mask.unsqueeze(0), torch.finfo(x.dtype).max)
        return x

m = M()
x = torch.randn(6)
torch.onnx.export(m, (x,), 'mask.onnx')