import torch
x = torch.ones(0).cuda()
print(x.float().max().item()) # -inf
print(x.int().max().item()) # 0