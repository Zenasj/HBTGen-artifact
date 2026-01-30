import torch
import torch.nn as nn

class DDS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x, mask):
        out = x[mask]
        out = self.relu(out)
        return out

model = DDS().eval().cuda()
x = torch.randn(1, 3, 4, 4).cuda()
y = torch.rand((1, 3, 4, 4), device="cuda") < 0.9
y = y.to(torch.int32)
inputs=(x, y)
model(*inputs)