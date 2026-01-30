import torch

a = torch.randn(3, 3)
b = torch.randn(3, 3)
c = torch.randn(3, 3)
flat_inp = (a, b, c)

def reduce_sub(flattened):
    init = 1
    for val in flattened:
        init -= val
    return init

def reduce_mul(flattened):
    init = 1
    for val in flattened:
        init *= val
    return init

def reduce_div(flattened):
    init = 1
    for val in flattened:
        init /= val
    return init

def test_func(reduce_func, inp):
    torch._dynamo.reset()
    return torch.compile(reduce_func, backend="eager")(inp)

# Ok
reduce_div(flat_inp)
# Ok
reduce_mul(flat_inp)
# Ok
reduce_sub(flat_inp)
# Error
test_func(reduce_sub, flat_inp)
# Error
test_func(reduce_mul, flat_inp)
# Error
test_func(reduce_div, flat_inp)

import torch

def fn(x):
    y = 1
    y -= x
    return y

x = torch.rand((4, 4))

fn(x)
torch.compile(fn, backend="eager")(x)