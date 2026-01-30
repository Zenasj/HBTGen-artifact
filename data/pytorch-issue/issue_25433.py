import torch
x = torch.rand(3, 4)
def foo(x):
    return x.type(torch.int8)
scr = torch.jit.script(foo)
print(scr(x))