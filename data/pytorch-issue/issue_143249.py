import torch
torch._dynamo.config.capture_scalar_outputs = True
def f(x, t):
    c = x.item()
    torch._check(c >= 0)
    torch._check(c < t.size(0))
    return torch.select(t, 0, c) + 1

out = torch.compile(f, fullgraph=True)(torch.tensor(3, dtype=torch.int64), torch.randn(4, 4))   
print(out)

import torch
torch._dynamo.config.capture_scalar_outputs = True
def f(x, t):
    c = x.item()
    torch._check_is_size(c)
    torch._check(c < t.size(0))
    return torch.select(t, 0, c) + 1

out = torch.compile(f, fullgraph=True)(torch.tensor(3, dtype=torch.int64), torch.randn(4, 4))   
print(out)

import torch
torch._dynamo.config.capture_scalar_outputs = True
def f(x, t):
    c = t.sum().clamp(min=0, max=4).to(torch.int64).item()
    torch._check(c >=0)
    torch._check(c < t.size(0))
    return torch.select(t, 0, c) + 1

out = torch.compile(f, fullgraph=True)(torch.tensor(3, dtype=torch.int64), torch.randn(4, 4))