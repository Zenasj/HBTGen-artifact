py
import torch
def fn(x):
    return torch.histc(x, bins=10, min=0, max=0.99)

x = torch.linspace(0, 0.99, 1001, dtype=torch.float32)
print(fn(x))
print(fn(x.cuda()))

tensor([101.,  99., 101.,  99., 100., 100., 100., 100., 100., 101.])
tensor([101.,  99., 101.,  99., 101.,  99., 100., 100., 100., 101.],
       device='cuda:0')