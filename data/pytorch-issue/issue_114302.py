import torch

@torch.compile(backend="inductor")
def foo(x):
  y = x.expand(2, *x.shape)
  y[0, 0] = 5
  return y

device = "cpu"
x = torch.arange(6, device=device)
r = foo(x)

print(x)
print(r)

# Expected (Eager CPU)
tensor([ 5,  1, 2, 3, 4, 5])
tensor([[ 5,  1, 2, 3, 4, 5],
        [ 5, 1, 2, 3, 4, 5]])

# Actual
tensor([ 5,  2,  4,  6,  8, 10])
tensor([[ 5,  2,  4,  6,  8, 10],
        [ 5,  2,  4,  6,  8, 10]])

import torch

@torch.compile(backend="inductor")
def foo(x):
    y = x.expand(3, *x.shape) #expand(1, *x.shape)
    y[0, 0].mul_(5)
    return x

device = "cuda"
x = torch.arange(6, device=device, dtype=torch.float16)
with torch.no_grad():
    r = foo(x)

print(x)
print(r)