import torch.nn.functional as F

import torch
from torch.nn import functional as F
x = torch.ones(3, 10)

y1 = F.pad(x, [1, 1, 1, 1])
y2 = F.pad(x, [0, 0, 0, 0])
y3 = x + 0

y1.unsqueeze_(0)
print(x.shape)  # 3, 10

y3.unsqueeze_(0)
print(x.shape)  # 3, 10

y2.unsqueeze_(0)
print(x.shape)  # 1, 3, 10
print(x is y2)  # True

y1.requires_grad = True
print(x.requires_grad)  # False
y3.requires_grad = True
print(x.requires_grad)  # False
y2.requires_grad = True
print(x.requires_grad)  # True