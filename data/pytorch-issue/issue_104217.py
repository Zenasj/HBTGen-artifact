import torch
a = torch.randn(1, 32, 32, 4, device="cuda")
z = torch.tensor([0], device="cuda")
b = torch.randn(33, 1, device="cuda")
idx0 = torch.randint(32, (33,), device="cuda")
idx1 = torch.randint(32, (33,), device="cuda")

@torch.compile
def f(a, z, b, idx0, idx1):
    a.index_put_((z, idx0, idx1), b, accumulate=True)

f(a, z, b, idx0, idx1)