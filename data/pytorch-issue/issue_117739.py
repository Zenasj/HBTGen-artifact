import torch
t1 = torch.rand((1000,768))
x1 = torch.matmul(t1, t1.T) # 1.34 ms ± 42.8 µs per loop
y1 = torch.matmul(t1, t1.T.detach().clone()) # 1.92 ms ± 48.2 µs per loop
assert x1.allclose(y1) # True

t2 = torch.rand((10000,768))
x2 = torch.matmul(t2, t2.T) # 270 ms ± 2.83 ms per loop
y2 = torch.matmul(t2, t2.T.detach().clone()) # 181 ms ± 2.45 ms per loop
assert x2.allclose(y2) # True