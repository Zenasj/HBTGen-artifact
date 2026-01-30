import torch

def fn(x):
    x=x.clamp(min=1.)*.1
    return x

x=torch.randn(4, device="cuda", dtype=torch.bfloat16)
scripted = torch.jit.script(fn)
fn(x)
with torch.jit.fuser("fuser2"):
    for _ in range(10):
        out = scripted(x)