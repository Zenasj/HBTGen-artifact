import torch

def baz(x):
    return torch.neg(x) + 3.14159

baz_traced = torch.jit.trace(baz, (torch.rand(5, 4, 3),))

@torch.jit.script
def foo(x):
    bar = torch.neg(torch.mm(x, torch.zeros(3, 4, 5)))
    return baz_traced(bar)

foo.save('test.out')

imported = torch.jit.load('test.out')
imported(torch.rand(3, 4))