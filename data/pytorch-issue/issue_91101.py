#!/usr/bin/env python

import sys
import torch
from torch._dynamo import optimize, config
config.cache_size_limit = 4
torch.manual_seed(0)

class Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sqrt(i.numel())
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * sqrt(result.numel())

@optimize("inductor")
def toy_example(x):
    return Op.apply(x).norm()

def test():
    for i in range(5):
        x = torch.randn(10)
        print(toy_example(x))

if __name__ == "__main__":
    msg = "Usage: python bug-2.py numpy|math"
    assert len(sys.argv) == 2, msg
    if sys.argv[1] == "numpy":
        from numpy import sqrt
    elif sys.argv[1] == "math":
        from math import sqrt
    else:
        raise  Exception(msg)
    test()