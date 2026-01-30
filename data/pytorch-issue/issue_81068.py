import torch
from torch._subclasses.fake_tensor import FakeTensorMode

with FakeTensorMode.push():
    x = torch.empty(1, device="cuda")
    y = torch.empty(1, device="cuda:0")
    x + y