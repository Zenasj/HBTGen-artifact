import torch.nn as nn

import torch
CS = torch.nn.CosineSimilarity(dim=1)
a = torch.randn(20,1204,41,41)
b = torch.randn(1204,1,1)
c = CS(a, b)
print(c.shape)