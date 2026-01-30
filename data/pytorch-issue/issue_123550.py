import torch
from enum import IntEnum, auto

class A(IntEnum):
    idx = 0

def test(t: torch.Tensor) -> torch.Tensor:
    return t[:, A.idx]

a = torch.rand((1, 15))
c_test = torch.compile(test)
print(c_test(a))

import torch
from enum import IntEnum, auto

class A(IntEnum):
    idx = 0

def test(t: torch.Tensor) -> torch.Tensor:
    return t[A.idx]

a = torch.rand((1, 15))
c_test = torch.compile(test)
print(c_test(a))

import torch
from enum import IntEnum, auto

class A:
    idx = 0

def test(t: torch.Tensor) -> torch.Tensor:
    return t[:, A.idx]

a = torch.rand((1, 15))
c_test = torch.compile(test)
print(c_test(a))