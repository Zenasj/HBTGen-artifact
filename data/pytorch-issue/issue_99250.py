import torch.nn as nn

py
import torch
from torch._subclasses import FakeTensorMode


with FakeTensorMode():
    input = torch.randn(3, 5)
    target = torch.empty(3, dtype=torch.long).random_(5)
    torch.nn.functional.cross_entropy(input, target, label_smoothing=0.5)