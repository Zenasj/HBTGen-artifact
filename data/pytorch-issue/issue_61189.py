import numpy as np

import torch
# Choose the `slow_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
model = torch.jit.script(model)

class ConvTest(torch.nn.Module):
    def __init__(self):
        super(ConvTest, self).__init__()
        conv_a_stride = (2, 1, 1)
        conv_b_stride = (1, 2, 2)
        stride = tuple(map(np.prod, zip(conv_a_stride, conv_b_stride)))
        self.conv = torch.nn.Conv3d(3, 2, (3, 5, 2), stride=stride, padding=(3, 2, 0), bias=False)

    def forward(self, x):
        return self.conv(x)

model = ConvTest()
model = torch.jit.script(model)

import torch.nn as nn
from typing import Callable
def set_attributes(self, params: List[object] = None) -> None:
    if params:
        for k, v in params.items():
            if k != "self":
                setattr(self, k, v)

class IdentityModel(torch.nn.Module):
    def forward(self, a):
        return a

class ResBlock(nn.Module):
    def __init__(
        self,
        branch1_conv: nn.Module = None,
        branch2: nn.Module = None,
        branch_fusion: Callable = lambda x, y: x + y,
    ) -> nn.Module:
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = self.branch_fusion(x, self.branch2(x))
        return x

model = ResBlock(None, IdentityModel())
model = torch.jit.script(model)

import torch.nn as nn
class IdentityModel(torch.nn.Module):
    def forward(self, a):
        return a

class Net(nn.Module):
    def __init__(self, basic_model):
        super().__init__()
        self.blocks = nn.ModuleList([basic_model, basic_model, basic_model])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
model = Net(IdentityModel())
model = torch.jit.script(model)