import torch.nn as nn

import torch
import functorch.experimental.control_flow as control_flow
def true_fn(x):
    return x.sin()

def false_fn(x):
    return x, x

def f(x, y):
    return control_flow.cond(y, true_fn, false_fn, [x])

f(torch.ones(3, 4), torch.tensor(False))

import torch
import functorch.experimental.control_flow as control_flow

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer", torch.ones(6, 4))

    def forward(self, x):
        def true_fn(x):
            self.buffer += 1
            return self.buffer.sum() + x.sum()

        def false_fn(x):
            return (x - 1).sum()

        return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [x])

mod_for_compile = torch.compile(Foo(), backend="eager", dynamic=True)
mod_for_compile(torch.ones(3, 4))