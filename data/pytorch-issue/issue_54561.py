import time
import torch


a = torch.eye(10000).to_sparse().coalesce()
t = time.time()
c = a.index_select(0, torch.arange(1000))
print(time.time() - t)

t = time.time()
b = []
for i in range(1000):
    b.append(a[i])
b = torch.stack(b)
print(time.time() - t)
print((b.to_dense() == c.to_dense()).all())

8.997999906539917
0.08900022506713867
tensor(True)