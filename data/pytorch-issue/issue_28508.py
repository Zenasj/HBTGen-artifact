import torch
print(torch.__version__)
mask = (torch.triu(torch.ones(3, 3)) == 1).transpose(0, 1)
print(mask)
mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
print(mask)