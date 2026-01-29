# (torch.rand(2, 2, dtype=torch.float32), torch.rand(2, 2, dtype=torch.float32), torch.rand(2, 2, dtype=torch.float32))
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

class MyModel(nn.Module):
    def forward(self, inputs):
        base, primal, tangent = inputs
        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            view = base.transpose(0, 1)
            view.mul_(dual)
            view *= 2  # Second in-place operation
            p1, d1 = fwAD.unpack_dual(base)
            return p1, d1

def my_model_function():
    return MyModel()

def GetInput():
    base = torch.rand(2, 2, dtype=torch.float32)
    primal = torch.rand(2, 2, dtype=torch.float32)
    tangent = torch.rand(2, 2, dtype=torch.float32)
    return (base, primal, tangent)

