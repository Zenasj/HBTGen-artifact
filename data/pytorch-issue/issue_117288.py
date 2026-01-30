import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from typing import cast

with FakeTensorMode():
    t = cast(FakeTensor, torch.empty([]))
print(torch.zeros(t))

t = torch.empty([])
print(torch.zeros(t))

t = torch.tensor(3)
print(torch.zeros(t))