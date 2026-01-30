import torch
a = torch.zeros(10000, 10000) # probability of drawing "1" is 0
print((torch.bernoulli(a) == 1).sum())

import torch
a = torch.zeros(10000, 10000, dtype=torch.float64) # probability of drawing "1" is 0
print((torch.bernoulli(a) == 1).sum())