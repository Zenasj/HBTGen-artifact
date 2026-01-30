import torch

def f(x):
    ctx = torch.set_grad_enabled(True)

    torch._dynamo.graph_break()

    with ctx:
        x = x + 1
    return x

opt_f = torch.compile(f, backend='eager')
opt_f(torch.randn(3, 3))
print("done!")