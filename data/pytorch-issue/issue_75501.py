import torch
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.logging_tensor import LoggingTensor

primal_f = torch.ones(2, 2, dtype=torch.float) 
primal_l = torch.ones(2, 2, dtype=torch.long)

tangent_f = torch.ones(2, 2, dtype=torch.float)
tangent_l = torch.ones(2, 2, dtype=torch.long)


def fn(x):
    return x ** 2

with fwAD.dual_level():
    # Float Primal and Long Tangent works
    dual_input = fwAD.make_dual(primal_f, tangent_l)
    o = fn(dual_input)
    print(fwAD.unpack_dual(o).tangent)

    # Long Primal and Long Tangent works
    dual_input = fwAD.make_dual(primal_l, tangent_l)
    o = fn(dual_input)
    print(fwAD.unpack_dual(o).tangent)

    # Long Primal and Float Tangent works
    dual_input = fwAD.make_dual(primal_l, tangent_f)
    o = fn(dual_input)
    print(fwAD.unpack_dual(o).tangent)