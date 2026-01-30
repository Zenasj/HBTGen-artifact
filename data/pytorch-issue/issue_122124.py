import torch

@torch.compile()
def f(sz, x):
    s0, s1 = sz.tolist()
    r0, r1 = torch.ops.aten.split_with_sizes.default(x, [s0, s1])
    return torch.zeros(r0)

N = 7312
S0 = 420
S1 = N - S0

f(torch.tensor([S0, S1]), torch.randn(N))