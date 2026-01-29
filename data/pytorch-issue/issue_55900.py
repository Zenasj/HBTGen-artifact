# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

class TestModule(nn.Module):
    def __init__(self, k, s):
        super(TestModule, self).__init__()
        self.k = [k, k]
        self.s = [s, s]
        self.d = [1, 1]
        self.value = 0.0

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h, pad_w = get_same_padding(ih, self.k[0], self.s[0], self.d[0]), get_same_padding(iw, self.k[1], self.s[1], self.d[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=self.value)
        return x

class MyModel(nn.Module):
    def __init__(self, k, s):
        super(MyModel, self).__init__()
        self.first = TestModule(k, s)
        self.second = TestModule(k, s)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

def my_model_function():
    return MyModel(k=3, s=1)

def GetInput():
    return torch.randn(1, 3, 224, 224, requires_grad=True)

