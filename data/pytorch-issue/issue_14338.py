import torch.nn as nn

import numpy as np

from torch import nn
from torch.autograd import Variable, Function

import onnx


class MyFunction(Function):

    @staticmethod
    def forward(ctx, x, y):
        x = x.clone()
        x.masked_fill_(y > 0, 0)
        return x*x + y

    @staticmethod
    def symbolic(graph, x, y):
        return graph.op("MyFunction", x, y)


class MyModule(nn.Module):
    def forward(self, x, y):
        # you can combine your ATen ops with standard onnx ones
        x = nn.ReLU()(x)
        return MyFunction.apply(x, y)


import torch.onnx
torch.onnx.export(MyModule(),
                  (Variable(torch.ones(3,4)), Variable(torch.ones(3,4))),
                  "output.onnx",
                  verbose=True)