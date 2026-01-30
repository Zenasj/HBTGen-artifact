import torch

def fn(nt):
    tmp1 = nt.clone()
    tmp2 = nt.clone()
    torch._dynamo.graph_break()
    return tmp1.clone(), tmp2.clone().values().detach()

compiled_fn =torch.compile(fn, backend="aot_eager")

values = torch.rand(6, 8, requires_grad=True, device="cpu", dtype=torch.float32)
offsets = torch.tensor([0, 2, 3, 6], device="cpu")
nt = torch.nested.nested_tensor_from_jagged(
  values, offsets, lengths=None, min_seqlen=1, max_seqlen=8
)
out, _ = compiled_fn(nt)
out.values().sum().backward()

import torch
from torch.testing._internal.two_tensor import TwoTensor

class Split(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.a, x.b

    @staticmethod
    def backward(ctx, ga, gb):
        return TwoTensor(ga.clone(), gb.clone())

def fn(nt):
    tmp1 = nt.clone()
    tmp2 = nt.clone()
    torch._dynamo.graph_break()
    return tmp1.clone(), Split.apply(tmp2.clone())[0].detach()

compiled_fn =torch.compile(fn, backend="aot_eager")

a = torch.tensor([1., 2.], requires_grad=True)
b = torch.tensor([1., 2.], requires_grad=True)
x = TwoTensor(a, b).requires_grad_(True)
out, _ = compiled_fn(x)
Split.apply(out)[0].sum().backward()