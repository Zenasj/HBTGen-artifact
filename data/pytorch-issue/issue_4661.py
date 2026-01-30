import torch.nn as nn

import torch
net = torch.nn.Linear(30000,1).cuda()
data = torch.ones(10, 30000).cuda()
for rep in range(1000):
    batch = torch.autograd.Variable(data, requires_grad=True)
    net(batch).norm(2).backward(create_graph=True)

for p in model.parameters():
    p.grad = None