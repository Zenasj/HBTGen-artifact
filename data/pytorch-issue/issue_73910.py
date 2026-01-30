import torch.nn as nn

import torch
net = torch.nn.Linear(1,1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1 )
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=1)

net = torch.nn.Linear(1,1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1 )
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1, total_steps=10)