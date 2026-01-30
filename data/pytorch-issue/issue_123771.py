import torch

def be1(gm, ex):
    print("backend 1")
    return gm

def be2(gm, ex):
    print("backend 2")

@torch.compile(backend=be1)
@torch.compile(backend=be2)
def fn(x):
    return x + 1

fn(torch.randn(3, 3))

import torch

@torch.no_grad
@torch.enable_grad
def f():
    print(torch.is_grad_enabled())

f()

@torch.compile
def f(x):
    x = x + 1
    with torch._dynamo.disable():
        x = x + 1
        x = f2(x)
    return x

@torch._dynamo.disable
def g(x):
    with torch._dynamo.enable():
        x = x + 1
        x = g2(x)
    return x

@torch.compile
def h(x):
    return g(x)

def a(x):
    return x + 1

def b(x):
    torch._dynamo.graph_break()
    return x + 1

@compile
def c(x):
    with disable():
        x = b(x)
        x = torch.sin(x)
    return x

@compile
@disable
def d(x):
    with enable():
        x = a(x)
        x = torch.cos(x)
    return x