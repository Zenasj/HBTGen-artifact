import torch

@torch.compile(backend="eager")
def f(x):
    return x * 2

f(torch.randn(2, 3))
f(torch.randn(2, 4))
with torch.compiler.set_stance("fail_on_recompile"):
    f(torch.randn(2, 6))