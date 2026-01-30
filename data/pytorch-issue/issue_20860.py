import torch

x = torch.randn(32, 60, 60)

torch.bmm(x, x) # no problem without cuda

x = x.cuda()

(x ** 2).sum()  # no problem with other cuda operations

torch.bmm(x, x) # crash here