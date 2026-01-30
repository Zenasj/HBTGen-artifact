import torch.nn as nn
import torch.nn.functional as F

3
import torch
import onnx
import onnx.utils
from torch import nn
import numpy as np


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = nn.Linear(3, 1)
        self.elu = nn.ELU()

    def forward(self, inp):
        return self.elu(self.linear(inp))

model = TestModel()
inp = torch.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
torch.onnx.export(model, inp, "model.onnx")

class ELU(nn.ELU):
    def __init__(self, alpha=1., inplace=False, scale=1.):
        super(nn.ELU, self).__init__()
        self.alpha=alpha
        self.inplace=inplace
        self.scale=scale

class myCELU(nn.Module):
    def __init__(self, alpha=1.):
        super(myCELU, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        # return F.celu(input, self.alpha, self.inplace)
        # max(0,x)+min(0,α∗(exp(x/α)−1))
        zeros = torch.zeros(input.shape)
        return torch.max(zeros, input) + torch.min(zeros, self.alpha * (torch.exp(input/self.alpha) - 1) )