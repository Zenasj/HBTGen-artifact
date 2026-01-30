#!/usr/bin/env python

import torch
from torch._dynamo import optimize, config
config.cache_size_limit = 4
torch.manual_seed(0)

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

@optimize("inductor")
def toy_example(x, dy):
    y = Exp.apply(x)
    y.backward(dy)
    return y

def test():
    for i in range(5):
        x = torch.randn(10, requires_grad=True)
        dy = torch.randn(10)
        toy_example(x, dy)
        print(x.grad.norm())

if __name__ == "__main__":
    test()