import torch

t = torch.tensor([1,1,1,1], device="mps")
a = t.cumsum(0)
b = t.cumsum(0, dtype=torch.int32)
c = t.cumsum(0, dtype=torch.int16)
d = t.cumsum(0, dtype=torch.int8)

print(a)
print(b)
print(c)
print(d)