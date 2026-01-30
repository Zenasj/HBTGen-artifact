import torch.nn as nn

import torch.export
import torch.nn

class M(torch.nn.Module):
    def forward(self, x):
        a, b = x.tolist()
        return x + a + b

torch.export.export(M(), [torch.tensor([1.0, 2.0])], {}, strict=False)