import torch.nn as nn

import torch
import numpy as np

class Cplx(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cplx', torch.from_numpy(1j*np.arange(10)))

    def forward(self, x):
        return x*self.cplx

cplx = torch.nn.DataParallel(Cplx().cuda())
x = torch.rand(2,10).cuda()
cplx(x)