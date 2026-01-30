import torch
d = torch.randint(0, 50000, size=(32, ), device=torch.device("mps"))
# d = torch.rand(size=(32, ), device=torch.device("mps"))

print(d)
print(d.dtype)
print("d1", d[1])
print("d1 + 1", d[1] + 1)
print("d2", d[2])
print("d2 + 1", d[2] + 1)

import torch
d = torch.randint(0, 50000, size=(32, ), device=torch.device("mps"))
# d = torch.rand(size=(32, ), device=torch.device("mps"))

print(d)
print(d.dtype)
print("d1", d[1])
print("d1 * 2", d[1] * 2)
print("d2", d[2])
print("d2 * 2", d[2] * 2)