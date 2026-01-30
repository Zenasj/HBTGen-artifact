# test.py
from typing import List
import torch

def fn002(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 1
    return x

def fn001(x):
    return fn002(x)

torch.compile(fn001, backend="eager")(torch.randn(1))