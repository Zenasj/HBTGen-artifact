import torch

x = torch.ones([100, 100, 100], device='cuda')

x = torch.tensor([10000, 256, 256, 3])
x = x.cuda()

x = torch.tensor([10000, 256, 256, 3])
x = x.to('cuda')