import torch

@torch.compile(fullgraph=True)
def fn(x):
    a = set(['a', 'b'])
    b = set(['a', 'c'])
    return a.intersection(b)


x = torch.randn(2, 3)
fn(x)