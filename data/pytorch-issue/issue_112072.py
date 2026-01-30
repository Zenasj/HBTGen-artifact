py
import torch

def f(x):
    y = x.clone()
    torch.set_grad_enabled(False)
    return y.clone()

f_compiled = torch.compile(f, backend='aot_eager', fullgraph=True)

x = torch.randn(3, requires_grad=True)
y = f(x)
print(torch.is_grad_enabled())
torch.set_grad_enabled(True)
y = f_compiled(x)
print(torch.is_grad_enabled())