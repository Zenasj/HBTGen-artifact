import torch
torch.set_default_device('cuda')

def f(a, b):
    return (a[0] * b[0] + a[1] * b[1])

x = [torch.randn(5) for _ in range(3)]
y = [torch.randn(5) for _ in range(3)]
torch.compile(lambda x, y: map(f, x, y), fullgraph=True)(x, y)