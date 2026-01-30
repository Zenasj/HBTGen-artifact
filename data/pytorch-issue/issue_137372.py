import torch

@torch.compile(fullgraph=True)
def f(x, y, z, w):
    y = torch.ops.aten.addmm(x, y, z)
    return y.view(-1).sin()

x = torch.randn(1308, requires_grad=True, device='cuda')
y = torch.randn(8, 256, requires_grad=True, device='cuda')
z = torch.randn(1308, 256, requires_grad=True, device='cuda').transpose(1, 0)
w = torch.randn(8, 1308, requires_grad=True, device='cuda')
with torch._dynamo.utils.maybe_enable_compiled_autograd(
    True, fullgraph=True, dynamic=False
):
    out = f(x, y, z, w)
    out.sum().backward()