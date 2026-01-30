import torch
import torch.nn as nn


class ModuleListTruncated(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])

    def forward(self, x):
        for fc in self.fcs[:1]:
            x = fc(x)
        return x


x = torch.rand(2, 1)
mod_truncated = ModuleListTruncated()

# torch.export.export(mod_truncated, (x,), strict=True)  # passes
torch.export.export(mod_truncated, (x,), strict=False)  # fails