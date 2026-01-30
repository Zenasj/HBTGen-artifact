import torch
import io

def fn(x):
    for i in range(1010):
        x = x.sin()
    return x

x = torch.rand((2, 2))
fn_s = torch.jit.trace(fn, (x,))
b = io.BytesIO()
torch.jit.save(fn_s, b)
b.seek(0)
gn_s = torch.jit.load(b)
print(fn_s.graph)