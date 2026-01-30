import torch
x = torch.tensor([0, 1, 2]).cuda()
x.topk(1)