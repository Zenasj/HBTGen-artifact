import torch.nn as nn

import torch
from torch import nn

model = nn.MultiheadAttention(128, 8, 0.1)
model = torch.jit.script(model)

q = torch.randn(64, 1, 128)
k = torch.randn(64, 1, 128)
v = torch.randn(64, 1, 128)

# prove that these are the correct shapes and args
_ = model(q, k, v)

torch.onnx.export(
    model,
    (q, k, v),
    "test.onnx",
)

import torch
from torch import nn


class MyNetwork(nn.Module):
    def forward(self, x, y=None):
        return x


model = MyNetwork()
model = torch.jit.script(model)

# this still works
x = torch.randn(8, 8)
y = model(x)

# this doesn't
torch.onnx.export(
    model,
    x,
    "test.onnx",
)

class Wrapper(nn.Module):
    def __init__(self, inner: MyNetwork):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x, torch.empty(()))


model = Wrapper(MyNetwork())