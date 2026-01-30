import torch

def f(x):
    torch._check(x.shape[0] == 1, lambda: "Failed")
    return x + 1

x = torch.compile(f, dynamic=True)(torch.zeros(2))

async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (s0, ), (1, ))
    return (Eq(s0, 1), )

import torch

def f(x):
    torch._check(x.shape[0] > 8 , lambda: "Failed")
    return x + 1

x = torch.compile(f, dynamic=True)(torch.zeros(2))