import torch

p = torch.tensor([2/10, 3/10, 5/10])
n = 10
out = torch.multinomial(p, n, replacement=True)
print(out) # tensor([1, 2, 1, 2, 2, 2, 1, 2, 0, 2])
 
# User needs to convert out (the outcomes of n trials) 
# to random variable using method as suggested here, 
# https://discuss.pytorch.org/t/trying-to-understand-the-torch-multinomial/71643/4
# This leads to unsound sampling from multinomial distribution
# without requiring user's further understanding and care
x = out.unique(return_counts=True)[1]
print(x) # tensor([1, 3, 6])

import torch
 
p = torch.tensor([2/10, 3/10, 5/10])
n = 3
out = torch.multinomial(p, n)
print(out) # tensor([1, 0, 2])
 
x = out.unique(return_counts=True)[1]
print(x) # tensor([1, 1, 1])