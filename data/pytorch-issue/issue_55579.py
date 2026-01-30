import torch
import torch.nn.functional as F

x = torch.tensor([1, 2, 3]).long().cuda()
F.one_hot(x, 4)