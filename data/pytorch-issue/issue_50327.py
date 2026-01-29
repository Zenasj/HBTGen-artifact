# torch.rand(1, 1, 1024, 2, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cplx_kaiming_uniform_(x, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    a = math.sqrt(1 + 2 * a * a)
    torch.nn.init.kaiming_uniform_(x[..., 0], a=a, mode=mode, nonlinearity=nonlinearity)
    torch.nn.init.kaiming_uniform_(x[..., 1], a=a, mode=mode, nonlinearity=nonlinearity)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cplx_conv = CplxConv(1, 1, 3)

    def forward(self, x):
        return self.cplx_conv(x)

class CplxConv(nn.Module):
    def __init__(self, inc, outc, kern):
        super(CplxConv, self).__init__()
        self.outc = outc
        self.padding = kern // 2
        self.weight = torch.nn.Parameter(torch.Tensor(outc, inc, kern, 2))
        cplx_kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        xre, xim = x[..., 0], x[..., 1]
        wre, wim = self.weight[..., 0], self.weight[..., 1]
        ww = torch.cat([wre, wim], dim=0)
        wr = F.conv1d(xre, ww, padding=self.padding)
        wi = F.conv1d(xim, ww, padding=self.padding)
        rwr, iwr = wr[:, :self.outc], wr[:, self.outc:]
        rwi, iwi = wi[:, :self.outc], wi[:, self.outc:]
        return torch.stack([rwr - iwi, iwr + rwi], dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 1024, 2, dtype=torch.float32)

