import torch
@torch.compile(backend="eager", fullgraph=True)
def f(t):
    xs = ["bar", "foo", "baz"]
    return t + xs.index("foo")

f(torch.randn(1))