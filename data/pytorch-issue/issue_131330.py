import torch
import torch.nn as nn

print(torch.equal(torch._C._nn.linear(t11, t12, t13), torch._C._nn.linear(t21, t22, t23)))
print(torch.equal(t11, t21))
print(torch.equal(t12, t22))
print(torch.equal(t13, t23))
False
True
True
True