import torch

input = torch.randn(128, 1, 1536, 49, 3, 3).cuda()
mask = torch.randn(1, 1536, 1536, 1, 3, 3).bool().cuda()
output = torch.masked_select(input, mask)