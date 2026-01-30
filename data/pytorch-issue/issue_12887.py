import torch

size = 3*1000*1000*1000
a = torch.LongTensor(size).random_(2)
(a == a).sum()

size = 3*1000*1000*1000
a = torch.ByteTensor(size).random_(2)
(a == 0).sum()