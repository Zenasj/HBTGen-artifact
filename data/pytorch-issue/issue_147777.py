import torch

@torch.compiler.allow_in_graph
def f(x):
    return x + 1
del f

def g(x):
    return x + 2

@torch.compile(fullgraph=True, backend="eager")
def fn(x):
    return g(x)

fn(torch.ones(1))