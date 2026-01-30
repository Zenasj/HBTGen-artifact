import torch

def division(Y, y):
    return (Y / y.to(Y.dtype).unsqueeze(-1))

Y, y = torch.randn(size=(1, 6, 128, 4, 32), dtype=torch.bfloat16, device='cuda'), torch.randn(size=(1, 6, 128, 4), dtype=torch.float32, device='cuda')

print("Y.mean()", Y.mean())
print("Y.std()", Y.std())
print("y.mean()", y.mean())
print("y.std()", y.std())

out_eager = division(Y, y)
out_compiled = torch.compile(division)(Y, y)

print(torch.allclose(out_eager, out_compiled))
print("diff", (out_eager - out_compiled).abs().max())
print("torch.version", torch.__version__)

def division(Y, y):
    return (Y / y.unsqueeze(-1)).to(Y.dtype)

import torch

def fn(discount):
    discount = discount.cumsum(2, dtype=torch.float32)
    return discount


discount = torch.randn(size=(1, 6, 128, 4), dtype=torch.float32, device='cuda')

print("discount.mean()", discount.mean())
print("discount.std()", discount.std())

out_eager = fn(discount)
out_compiled = torch.compile(fn)(discount)

print(torch.allclose(out_eager, out_compiled))
print("diff:", torch.max(torch.abs(out_eager - out_compiled)))