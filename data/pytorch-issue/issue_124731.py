py
import torch
import torch
import torch.autograd.forward_ad as fwAD

def f(x):
    return 4312491 * x

device = "cpu"

with torch._subclasses.fake_tensor.FakeTensorMode():
    with fwAD.dual_level():
        x = torch.randn(3, device=device)
        y = torch.ones_like(x)
        dual = fwAD.make_dual(x, y)
        f(dual)