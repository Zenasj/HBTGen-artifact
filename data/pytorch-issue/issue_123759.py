import torch

def f(x):
    return x + 2

torch._export.aot_compile(f, (torch.randn(3),))

import torch


def compile_fn(x):
    return x + 2


torch._export.aot_compile(compile_fn, (torch.randn(3),))

ep = torch.export.export(mod)
print(aot_compile(ep, ...))
print(serialize(ep, ...))
ep2 = run_pass1(ep)
ep3 = run_pass2(ep2)
...