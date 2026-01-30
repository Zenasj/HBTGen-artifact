import torch.nn as nn

import torch
from torch.nn.modules import CrossEntropyLoss

dev = 'mps'

pred = torch.randn(3, 5, requires_grad=True).to(dev)
target = torch.ones(3, dtype=torch.long).to(dev)

loss = CrossEntropyLoss()
output = loss(pred, target)
output.backward()
print(output)

import torch
from torch.nn.modules import CrossEntropyLoss

dev = 'mps'

pred = torch.randn(3, 5, requires_grad=True).to(dev)
target = torch.ones(3, dtype=torch.long)
target = torch.eye(5)[target].to(dev)

loss = CrossEntropyLoss()
output = loss(pred, target)
output.backward()
print(output)