import torch.nn as nn

import torch

x = torch.randn(10,2048).cuda()
y = torch.randn(10,2048).cuda()

print(torch.nn.CosineSimilarity()(x,y).mean())

x= x.half()
y =y.half()

print(torch.nn.CosineSimilarity()(x,y).mean())
print(torch.diagonal(x @ y.T).mean())

x = torch.nn.functional.normalize(x)
y = torch.nn.functional.normalize(y)

print(torch.nn.CosineSimilarity()(x,y).mean())
print(torch.diagonal(x @ y.T).mean())