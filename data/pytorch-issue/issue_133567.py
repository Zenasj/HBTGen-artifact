import torch


@torch.compile
def f(x):
    return torch.matmul(x, x).sin()

x = torch.randn(4, 4, requires_grad=True)
with torch._dynamo.utils.maybe_enable_compiled_autograd(True):
    out = f(x)
    out.sum().backward()