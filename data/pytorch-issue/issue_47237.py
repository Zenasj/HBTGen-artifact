import torch

a = torch.zeros(8, 1, 128, 1024, 1024)
a.cuda().sum(1)