import torch

def true_fn(x):
    return x - x.cos()

def false_fn(x):
    return x + x.sin()

def foo(x):
    return cond(x.shape[0] == 4, true_fn, false_fn, [x])
gm = make_fx(foo, tracing_mode='symbolic')(torch.ones(3, 4))