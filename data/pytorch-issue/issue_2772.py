import torch

x = torch.LongTensor(10).random_(10)
x = torch.autograd.Variable(x)
y = torch.mean(x)