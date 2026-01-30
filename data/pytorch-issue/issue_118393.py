import torch.nn as nn

import torch

def f(x):
    return x[x > 0]

jf = torch.jit.trace(f, torch.tensor(2., device="cuda"))

import torch
import torch.nn.functional as F
from torch import nn
import copy

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        x = copy.deepcopy(inputs) # RuntimeError: NYI: Named tensors are not supported with the tracer
        x = F.relu(x)
        return x

model = Net()
images = torch.randn(8, 28, 28)
torch.jit.trace(model, images)