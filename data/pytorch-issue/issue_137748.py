import torch.nn as nn

import torch 
class TestModel(torch.nn.Module):
    def forward(self, x):
        return x.shape[1]

from torch.export import Dim

model = TestModel().cuda()
x = torch.rand(2, 6, 6).cuda()
args = (x,)

bs = Dim("bs", max=128)
ep = torch.export.export(
    model, args, dynamic_shapes={"x": {1: 2 * bs}}, strict=False
)

aot_model = torch._inductor.aot_compile(ep.module(), args)

import torch
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y=None):
        return x + y.reshape(x.shape)

model = TestModule().cuda()
x = torch.ones(4, 4).cuda()
y = torch.rand(2, 8).cuda()

model(x, y)
from torch.export import Dim

d0 = Dim("d0")
d1 = Dim("d1")

ep = torch.export.export(
    model, (x, y), dynamic_shapes={"x": None, "y": {0: d0, 1: d1}}, strict=False
)

ep = export(
    model, (x, y), dynamic_shapes={"x": None, "y": {0: Dim.AUTO, 1: Dim.AUTO}}, strict=False
)
print(ep)

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y=None):
        return x + y.reshape(x.shape)

model = TestModule().cuda()
x = torch.ones(4, 4).cuda()
y = torch.rand(2, 8).cuda()

model(x, y)
from torch.export import Dim

_ = torch.export.export(
    model,
    (x, y),
    dynamic_shapes={"x": None, "y": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC}},
    strict=False,
)