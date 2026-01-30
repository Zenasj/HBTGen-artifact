import functools
import torch


def decorator(func):
    @functools.wraps(func)  # <--- works if one comment this line
    def helper(*args):
        return func(*args)
    return helper


def g(x):
    @decorator
    def h():
        return x * 100
    return h


def run(h):
    return h()


@torch.compile(fullgraph=True)
def fn(x):
    h = g(x)
    return run(h)


x = torch.randn(1)
y = fn(x)
print(x, y)