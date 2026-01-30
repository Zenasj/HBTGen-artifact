import torch.nn as nn

import torch
import torch.nn.functional as F
import sys

INT32_MAX = 2 ** 31 - 2

sys.setrecursionlimit(INT32_MAX)
x = torch.randn(400, device="cpu")


y = 0

@torch.compile # Comment this out to let CPython 3.11 optimize calls
def fn(x):
    def inner():
        global y
        if y < INT32_MAX - 1:
            print(y)
            y += 1
            return inner()
    inner()
    return F.softshrink(x)

fn(x)