import torch
import torch.nn as nn

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, ys):
        a = torch.sin(x)
        b = torch.cos(ys[0])
        c = torch.cos(ys[1])
        return (x, [b ,c])

mod = MockModule().cuda()

def fn(x, ys):
    from torch.utils import checkpoint
    return checkpoint.checkpoint(mod, x, ys)

x = torch.randn(4, 4).cuda()
y = torch.randn(4, 4).cuda()
z = torch.randn(4, 4).cuda()
ref = fn(x, [y, z])
opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
res = opt_fn(x, [y, z])
torch.testing.assert_close(res, ref)