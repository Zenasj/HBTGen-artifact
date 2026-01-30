import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

dim = torch.export.Dim("batch", min=1, max=6)
ep = torch.export.export(
    Model(),
    (torch.randn(2, 3), torch.randn(2, 3)),
    dynamic_shapes=[{0: dim}, {0: dim}],
)