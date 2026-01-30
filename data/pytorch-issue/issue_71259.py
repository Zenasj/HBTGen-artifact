import torch

import torch.nn as nn

import torch.fx.experimental.optimization as optimization

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv = nn.Conv2d(32, 64, 3, stride=2)
        self.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

x = torch.randn([1, 32, 50, 50])

model = M().eval()

'''
# jit path
with torch.no_grad():
    traced = torch.jit.trace(model, x).eval()
    traced = torch.jit.freeze(traced)
'''

# FX path
fused_model = optimization.fuse(model)