import torch
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def f(sz, x):
    s0, s1 = sz.tolist()
    r0, r1 = torch.ops.aten.split_with_sizes.default(x, [s0, s1])
    return torch.ops.aten.sort.default(r1)

N = 7312
S0 = 420
S1 = N - S0

f(torch.tensor([S0, S1]), torch.randn(N))

import torch
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True

torch.library.define("ezyang::split_with_sizes_and_clone", "(Tensor input, SymInt[] sizes) -> Tensor[]")

def split_with_sizes_and_clone(input, sizes):
    return [t.clone() for t in torch.ops.aten.split_with_sizes.default(input, sizes)]

torch.library.impl("ezyang::split_with_sizes_and_clone", "default", split_with_sizes_and_clone)

@torch.library.impl_abstract("ezyang::split_with_sizes_and_clone")
def split_with_sizes_and_clone_abstract(input, sizes):
    # TODO: I'm lazy
    rs = torch.ops.aten.split_with_sizes.default(input, sizes)
    return [input.new_empty(r.size()) for r in rs]

@torch.compile()
def f(sz, x):
    s0, s1 = sz.tolist()
    r0, r1 = torch.ops.ezyang.split_with_sizes_and_clone.default(x, [s0, s1])
    return torch.ops.aten.sort.default(r1)

N = 7312
S0 = 420
S1 = N - S0

f(torch.tensor([S0, S1]), torch.randn(N))