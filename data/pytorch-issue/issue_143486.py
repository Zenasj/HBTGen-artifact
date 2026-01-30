import torch
a = torch.ones((4096*8, 100000), dtype=torch.float, device='cuda')
a /= 100000
c = a.cumsum(dim=-1)
print(c[0,-5:])

tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], device='cuda:0')

import torch
b = torch.ones((2*4096*8, 100000), dtype=torch.float, device='cuda')
b /= 100000
d = b.cumsum(dim=-1)
print(d[0,-5:])

tensor([0.6729, 0.6729, 0.6729, 0.6730, 0.6730], device='cuda:0')

print(b[0, 32702:32702+4])
print(d[0, 32702:32702+4])

tensor([1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05], device='cuda:0')
tensor([3.2703e-01, 3.2704e-01, 1.0000e-05, 2.0000e-05], device='cuda:0')

b = torch.ones((2*4096*8, 100000), dtype=torch.float, device='cuda')
b /= 100000
chunk = 2**30 // b.shape[-1]
for i in range(0, b.shape[0], chunk):
    end = min(i + chunk, b.shape[0])
    b[i:end, :].cumsum_(dim = -1)