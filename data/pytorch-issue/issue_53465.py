import torch

v1 = (2**31) - 1
for i in range(200):
    v2 = torch.Tensor([v1]).item()
    print(i, v1, v2)
    v1 -= 1

for i in range(2**31 - 10, 2**31 -1):
    a = torch.FloatTensor([1])
    a[0] = i
    print(i, a.item())

v1 = (2**31) - 1
v2 = torch.Tensor([v1])
print(v1, v2.item(), v2.to(torch.int32).item())