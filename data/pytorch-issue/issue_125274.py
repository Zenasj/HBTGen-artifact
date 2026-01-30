import torch

def g(ctx):
    torch._dynamo.graph_break()
    return ctx

def f(x):
    x = x + 1
    ctx = g(torch.set_grad_enabled(True))
    with ctx:
        x = x + 1
    return x

opt_f = torch.compile(f, backend='eager')
opt_f(torch.randn(3, 3))
print("done!")