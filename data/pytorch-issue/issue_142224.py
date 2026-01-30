import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.gelu(x)
        return x


m = Model().cuda()
x = torch.randn(64, 64).cuda()

with torch.no_grad():
    m = m.eval()
    c_m = torch.compile(m)  # if backend="cudagraphs", there is no inconsistency

    y = m(x)
    c_y = c_m(x)

    print(torch.allclose(y, c_y))
    print(torch.max(torch.abs(y - c_y)))