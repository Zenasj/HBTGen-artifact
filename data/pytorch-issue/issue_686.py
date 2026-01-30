import torch

print(torch.cuda.is_available())

x = torch.rand(5, 3)
y = torch.rand(5, 3)
x = x.cuda()
y = y.cuda()
x + y