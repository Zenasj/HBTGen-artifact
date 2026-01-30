import torch.nn as nn

import torch
from torch import nn



y = torch.randn(100).to(device='mps')
torch.manual_seed(0)
model = nn.Dropout(.3)
result1 = model(y)
torch.manual_seed(0)
model = nn.Dropout(.3)
result2 = model(y)
print(torch.all(torch.eq(result1,result2)))

y = torch.randn(100).to(device='cpu')
torch.manual_seed(0)
model = nn.Dropout(.3)
result1 = model(y)
torch.manual_seed(0)
model = nn.Dropout(.3)
result2 = model(y)
print(torch.all(torch.eq(result1,result2)))