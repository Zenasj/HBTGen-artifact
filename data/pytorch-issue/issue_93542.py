import torch

d1 = 2
d2 = 197
d3 = 256

torch.manual_seed(20)
a = torch.randn(d1, d2, d3)
b = a.sum((0, 1))
print(b[0].item())

ref = torch.zeros(d3, )
for k in range(d3):
  b = a[:, :, k]
  for i in range(d1):
      for j in range(d2):
        ref[k] += b[i][j]
print(ref[0].item())

ref = torch.zeros(d3, )
for k in range(d3):
  b = a[:, :, k]
  for i in range(d2):
      for j in range(d1):
        ref[k] += b[j][i]
print(ref[0].item())