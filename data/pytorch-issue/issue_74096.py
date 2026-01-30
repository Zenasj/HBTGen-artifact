import torch.nn as nn

from torch import nn


class A:
    def __init__(self):
        super().__init__()
        self.a = True


class B(A, nn.Module):
    def __init__(self):
        super().__init__()
        self.b = True


class C(nn.Module, A):
    def __init__(self):
        super().__init__()
        self.c = True


b = B()
c = C()

print(b.b)
print(b.a)

print(c.c)
print(c.a)