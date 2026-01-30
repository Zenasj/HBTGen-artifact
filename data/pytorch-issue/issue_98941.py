import torch
import torch._dynamo

def f(x, y):
    return x + y

def forward(x, y):
    return forward2(x, y)

def forward2(x, y):
    if x.size(0) < 128:
        x = x * 2
    else:
        x = x * 3
    r = f(x, y)
    r = r * y
    return r

def woof():
    fn_compiled = torch.compile(forward, dynamic=True)
    x = torch.randn(32, device='cuda')
    y = torch.randn(32, device='cuda')
    print(fn_compiled(x, y))

woof()