import numpy as np

import torch
from torch.autograd import Function

class Foo(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def forward(ctx, gO):
        return gO.clone()

def get_out():
    inp = torch.rand(2, requires_grad=True)

    # The python function is first so that it runs
    # last in the backward pass
    right = Foo.apply(inp)

    # An op that creates new memory
    left1 = inp.clone()
    # An op that saves its input
    left2 = left1 ** 2

    # Inplace modify so that the backward for
    # left2 always raises an error
    left1 += 1

    # An op that takes both side as input.
    # After running, both side's last op will be in
    # the ready queue
    # And the op for left will run first as it was
    # executed last during the forward
    out = left2 + right

    return out

# Nothing should be global variables here as, from what
# I can see, python leaks all the global objects
get_out().sum().backward()