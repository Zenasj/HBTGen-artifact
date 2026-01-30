import torch

@torch.compile
def f(x):
    tmp1 = x.sin()
    print("graph break!")
    return tmp1.sin()

x = torch.randn(4, requires_grad=True)
out = f(x)
with torch._dynamo.compiled_autograd.enable(torch.compile):
    out.sum().backward()