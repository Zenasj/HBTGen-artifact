import torch
import torch.autograd.forward_ad as fwAD

base = torch.randn(2, 2)
view = base.transpose(0, 1)

primal = torch.randn(2, 2)
tangent = torch.randn(2, 2)

with fwAD.dual_level():
    dual = fwAD.make_dual(primal, tangent)
    view.mul_(dual)

    p, d = fwAD.unpack_dual(base)
    print(p, d)
    view *= 2
    p1, d1 = fwAD.unpack_dual(base)
    print(p1, d1)