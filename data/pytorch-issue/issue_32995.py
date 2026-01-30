import torch
x = torch.zeros(64, 64).cuda()
x = torch.ByteTensor(x)
print("Hello world")