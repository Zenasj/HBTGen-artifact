import torch.nn as nn

import torch

m = torch.nn.Linear(2, 2)
m_ = torch.compile(m, backend="inductor")

inp = torch.randn(2)

# Run with no_grad, so we create an inference graph.
with torch.no_grad():
    m_(inp)

# Oops - we should recompile, but we re-use the inference graph!
out = m_(inp)
# Errors:
# RuntimeError: mm(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.
# Why? inductor expects inference graphs to be run with torch.is_grad_enabled == False (because we must have guarded on it).
# But we didn't!
out.sum().backward()
print(m_.weight.grad)

import torch

def f(x):
    if x.requires_grad:
        return x + 1
    else:
        return x + 2

x = torch.ones(2, requires_grad=True)

f_compiled = torch.compile(f)
with torch.no_grad():
    # prints [2, 2]]
    print(f(x))
    # prints [2, 2]]
    print(f_compiled(x))

# prints [3, 3]]
print(f(x.detach()))
# prints [2, 2]]
print(f_compiled(x.detach()))