import torch
x = torch.rand(10).cuda()
try:
  y = x[torch.tensor([11])]
  print(y)
except RuntimeError as err:
  print("Error:", err)
z = x[torch.tensor([1])]
print(z)