import torch


def foo(x, y):
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    x = x + y
    return x

scripted = torch.jit.script(foo)
scripted.save('foo.zip')

loaded = torch.jit.load('foo.zip')
loaded(torch.rand(3, 4), torch.rand(4, 5))