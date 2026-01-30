import torch
import torch.autograd.forward_ad as fwAD

bug = True

class Add1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not bug:
            return x + 1
        else:
            return object(), x + 1

    @staticmethod
    def jvp(ctx, gx):
        if not bug:
            return gx
        else:
            return None, gx

fn = Add1.apply

primal  = torch.tensor([1,2,3], dtype=torch.float, requires_grad=True)
tangent = torch.tensor([4,5,6], dtype=torch.float)

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)

    if not bug:
        dual_output = fn(dual_input)
    else:
        _, dual_output = fn(dual_input)

    jvp = fwAD.unpack_dual(dual_output).tangent
    print(dual_input)
    print(dual_output)
    print(jvp)